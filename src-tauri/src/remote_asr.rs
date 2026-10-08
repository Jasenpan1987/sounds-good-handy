//! Speech-to-text on a remote server that speaks the OpenAI transcription API
//! (`POST /v1/audio/transcriptions`, multipart, bearer key), e.g. vLLM serving
//! Qwen3-ASR on a GPU instance.
//!
//! This is the one engine that runs no model on this machine, for Macs without
//! the memory to keep a local model loaded. Unlike the local engines, the
//! custom words go to the server as the `prompt` field: Qwen3-ASR reads it as
//! context, and a measured test set showed it fixes terms such as "merge" that
//! come back as Chinese homophones without it.

use std::io::Cursor;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{anyhow, Result};
use hound::{WavSpec, WavWriter};
use log::{debug, info};
use reqwest::multipart::{Form, Part};
use reqwest::{StatusCode, Url};
use serde::Deserialize;

const TRANSCRIPTIONS_PATH: &str = "/v1/audio/transcriptions";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
// Longer than reqwest's 90 s default so a connection warmed at the start of a
// dictation is still open when the recording is sent; a new TLS handshake to a
// distant server costs several hundred milliseconds.
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(240);

/// Everything one request needs, resolved from settings by the caller.
#[derive(Debug, Clone)]
pub struct RemoteAsrRequest {
    /// Base URL (`https://host`) or the full transcriptions endpoint.
    pub url: String,
    pub model: String,
    /// File holding the bearer key; `~/` is expanded.
    pub key_file: String,
    /// Terms the server should expect, comma-joined into `prompt`.
    pub custom_words: Vec<String>,
    /// Handy language code, or "auto".
    pub language: String,
}

#[derive(Deserialize)]
struct TranscriptionResponse {
    text: String,
}

fn client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .pool_idle_timeout(POOL_IDLE_TIMEOUT)
            .build()
            .expect("a client with only timeouts set always builds")
    })
}

/// Runs `fut` on Tauri's async runtime and waits for it on the calling thread.
/// `transcribe` is synchronous and is called from inside that runtime, where
/// `block_on` would panic; the long-lived runtime also keeps the client's
/// pooled connections alive between dictations.
fn wait_on_runtime<F, T>(fut: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    tauri::async_runtime::spawn(async move {
        let _ = tx.send(fut.await);
    });
    rx.recv()
        .map_err(|_| anyhow!("The speech server request was dropped"))?
}

/// Accepts `https://host`, `https://host/v1`, or the full endpoint.
pub fn transcriptions_url(base: &str) -> Result<Url> {
    let trimmed = base.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(anyhow!(
            "No speech server URL is set. Add it in Settings → Advanced → Speech server."
        ));
    }
    let full = if trimmed.ends_with(TRANSCRIPTIONS_PATH) {
        trimmed.to_string()
    } else if trimmed.ends_with("/v1") {
        format!("{trimmed}/audio/transcriptions")
    } else {
        format!("{trimmed}{TRANSCRIPTIONS_PATH}")
    };
    Url::parse(&full).map_err(|e| anyhow!("The speech server URL '{trimmed}' is not valid: {e}"))
}

fn expand_home(path: &str) -> PathBuf {
    match (path.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => PathBuf::from(path),
    }
}

fn read_key(key_file: &str) -> Result<String> {
    let path = expand_home(key_file.trim());
    let key = std::fs::read_to_string(&path).map_err(|e| {
        anyhow!(
            "Could not read the speech server key from {}: {e}",
            path.display()
        )
    })?;
    let key = key.trim().to_string();
    if key.is_empty() {
        return Err(anyhow!(
            "The speech server key file {} is empty",
            path.display()
        ));
    }
    Ok(key)
}

/// 16 kHz mono 16-bit WAV, the format the recorder already produces and the
/// one Qwen3-ASR consumes; compressing it saved no time in testing.
pub fn encode_wav(samples: &[f32]) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buffer = Cursor::new(Vec::with_capacity(44 + samples.len() * 2));
    {
        let mut writer = WavWriter::new(&mut buffer, spec)?;
        for sample in samples {
            writer.write_sample((sample * i16::MAX as f32) as i16)?;
        }
        writer.finalize()?;
    }
    Ok(buffer.into_inner())
}

/// Handy language code → the server's `language` field. `None` for "auto":
/// Qwen3-ASR forces its output language when one is given, which breaks
/// mixed Chinese-English speech.
fn request_language(language: &str) -> Option<String> {
    match language {
        "" | "auto" => None,
        code if code.starts_with("zh") => Some("zh".to_string()),
        code => Some(code.to_string()),
    }
}

fn prompt_from(custom_words: &[String]) -> Option<String> {
    let words: Vec<&str> = custom_words
        .iter()
        .map(|w| w.trim())
        .filter(|w| !w.is_empty())
        .collect();
    (!words.is_empty()).then(|| words.join(", "))
}

fn describe_failure(status: StatusCode, body: &str) -> String {
    let detail: String = body.trim().chars().take(300).collect();
    let hint = match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
            "the server rejected the key; check the key file"
        }
        StatusCode::NOT_FOUND => "check the URL and the model name",
        StatusCode::PAYLOAD_TOO_LARGE => "the recording is too long for the server",
        StatusCode::BAD_GATEWAY | StatusCode::SERVICE_UNAVAILABLE | StatusCode::GATEWAY_TIMEOUT => {
            "the server is stopped or still starting"
        }
        _ => "the server returned an error",
    };
    if detail.is_empty() {
        format!("Speech server answered HTTP {status}: {hint}")
    } else {
        format!("Speech server answered HTTP {status}: {hint} ({detail})")
    }
}

fn describe_reqwest_error(url: &Url, error: &reqwest::Error) -> String {
    let host = url.host_str().unwrap_or("the speech server");
    if error.is_timeout() {
        format!("{host} did not answer in time")
    } else if error.is_connect() {
        format!("Could not connect to {host}; the server may be stopped")
    } else {
        format!("Request to {host} failed: {error}")
    }
}

pub fn transcribe(request: &RemoteAsrRequest, samples: &[f32]) -> Result<String> {
    let url = transcriptions_url(&request.url)?;
    let key = read_key(&request.key_file)?;
    let wav = encode_wav(samples)?;
    let model = request.model.trim().to_string();
    let prompt = prompt_from(&request.custom_words);
    let language = request_language(&request.language);
    debug!(
        "Speech server request: {} bytes of WAV, model '{}', prompt terms {}, language {:?}",
        wav.len(),
        model,
        request.custom_words.len(),
        language
    );

    let started = std::time::Instant::now();
    let text = wait_on_runtime(async move {
        let mut form = Form::new()
            .part(
                "file",
                Part::bytes(wav)
                    .file_name("audio.wav")
                    .mime_str("audio/wav")?,
            )
            .text("response_format", "json")
            .text("temperature", "0");
        if !model.is_empty() {
            form = form.text("model", model);
        }
        if let Some(prompt) = prompt {
            form = form.text("prompt", prompt);
        }
        if let Some(language) = language {
            form = form.text("language", language);
        }

        let response = client()
            .post(url.clone())
            .bearer_auth(key)
            .multipart(form)
            .send()
            .await
            .map_err(|e| anyhow!(describe_reqwest_error(&url, &e)))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| anyhow!(describe_reqwest_error(&url, &e)))?;
        if !status.is_success() {
            return Err(anyhow!(describe_failure(status, &body)));
        }
        let parsed: TranscriptionResponse = serde_json::from_str(&body)
            .map_err(|e| anyhow!("Speech server sent a response without text: {e}"))?;
        Ok(parsed.text)
    })?;
    info!(
        "Speech server round trip took {:.2}s",
        started.elapsed().as_secs_f64()
    );
    Ok(text.trim().to_string())
}

/// Opens the TLS connection in the background while the user is still
/// speaking, so the recording isn't charged for the handshake. Any answer,
/// even an error status, leaves a pooled connection behind.
pub fn warm_up(base_url: &str) {
    let Ok(endpoint) = transcriptions_url(base_url) else {
        return;
    };
    let mut health = endpoint;
    health.set_path("/health");
    tauri::async_runtime::spawn(async move {
        match client().get(health.clone()).send().await {
            Ok(response) => debug!("Speech server warm-up: HTTP {}", response.status()),
            Err(e) => debug!(
                "Speech server warm-up failed: {}",
                describe_reqwest_error(&health, &e)
            ),
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn transcriptions_url_accepts_host_v1_or_full_endpoint() {
        let want = "https://asr.example.com/v1/audio/transcriptions";
        for base in [
            "https://asr.example.com",
            "https://asr.example.com/",
            "https://asr.example.com/v1",
            want,
            " https://asr.example.com/v1/audio/transcriptions/ ",
        ] {
            assert_eq!(transcriptions_url(base).unwrap().as_str(), want, "{base}");
        }
        assert!(transcriptions_url("").is_err());
        assert!(transcriptions_url("not a url").is_err());
    }

    #[test]
    fn auto_and_empty_send_no_language_and_chinese_variants_collapse() {
        assert_eq!(request_language("auto"), None);
        assert_eq!(request_language(""), None);
        assert_eq!(request_language("zh-Hans").as_deref(), Some("zh"));
        assert_eq!(request_language("zh-Hant").as_deref(), Some("zh"));
        assert_eq!(request_language("en").as_deref(), Some("en"));
    }

    #[test]
    fn prompt_joins_trimmed_words_and_skips_blanks() {
        let words = vec![" UVAP ".to_string(), "".to_string(), "merge".to_string()];
        assert_eq!(prompt_from(&words).as_deref(), Some("UVAP, merge"));
        assert_eq!(prompt_from(&[]), None);
    }

    #[test]
    fn wav_is_16k_mono_16_bit_with_one_sample_per_input() {
        let wav = encode_wav(&[0.0, 0.5, -0.5, 1.0]).unwrap();
        let reader = hound::WavReader::new(Cursor::new(wav)).unwrap();
        let spec = reader.spec();
        assert_eq!(
            (spec.channels, spec.sample_rate, spec.bits_per_sample),
            (1, 16000, 16)
        );
        let samples: Vec<i16> = reader.into_samples().map(|s| s.unwrap()).collect();
        assert_eq!(samples, vec![0, 16383, -16383, i16::MAX]);
    }

    #[test]
    fn key_is_trimmed_and_missing_or_empty_files_are_errors() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "  secret-key  ").unwrap();
        assert_eq!(
            read_key(file.path().to_str().unwrap()).unwrap(),
            "secret-key"
        );

        let empty = tempfile::NamedTempFile::new().unwrap();
        assert!(read_key(empty.path().to_str().unwrap()).is_err());
        assert!(read_key("/nonexistent/sounds-good/key").is_err());
    }

    /// Serves exactly one request on a local socket, answers with `reply`, and
    /// hands back the raw request so the test can check what was sent.
    async fn one_shot_server(reply: &'static str) -> (String, tokio::task::JoinHandle<String>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut raw = Vec::new();
            let mut chunk = [0u8; 8192];
            loop {
                let n = socket.read(&mut chunk).await.unwrap();
                raw.extend_from_slice(&chunk[..n]);
                let text = String::from_utf8_lossy(&raw);
                if let Some(head_end) = text.find("\r\n\r\n") {
                    let length = text[..head_end]
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or(0);
                    if raw.len() >= head_end + 4 + length {
                        break;
                    }
                }
                if n == 0 {
                    break;
                }
            }
            socket.write_all(reply.as_bytes()).await.unwrap();
            String::from_utf8_lossy(&raw).into_owned()
        });
        (base, handle)
    }

    fn request_for(base: String, key_file: &tempfile::NamedTempFile) -> RemoteAsrRequest {
        RemoteAsrRequest {
            url: base,
            model: "Qwen/Qwen3-ASR-1.7B".to_string(),
            key_file: key_file.path().to_str().unwrap().to_string(),
            custom_words: vec!["UVAP".to_string(), "merge".to_string()],
            language: "auto".to_string(),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn sends_key_model_prompt_and_audio_and_returns_trimmed_text() {
        let body =
            r#"{"text":" 你去把这个PR给我merge一下。 ","usage":{"type":"duration","seconds":3}}"#;
        let reply: &'static str = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            )
            .into_boxed_str(),
        );
        let (base, server) = one_shot_server(reply).await;
        let mut key_file = tempfile::NamedTempFile::new().unwrap();
        writeln!(key_file, "test-key").unwrap();
        let request = request_for(base, &key_file);

        let text = tokio::task::spawn_blocking(move || transcribe(&request, &[0.0; 1600]))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(text, "你去把这个PR给我merge一下。");

        let raw = server.await.unwrap();
        assert!(raw.starts_with("POST /v1/audio/transcriptions "), "{raw}");
        assert!(raw
            .to_ascii_lowercase()
            .contains("authorization: bearer test-key"));
        for field in ["file", "model", "prompt", "response_format", "temperature"] {
            assert!(
                raw.contains(&format!("name=\"{field}\"")),
                "missing {field}"
            );
        }
        assert!(
            !raw.contains("name=\"language\""),
            "auto must send no language"
        );
        assert!(raw.contains("Qwen/Qwen3-ASR-1.7B"));
        assert!(raw.contains("UVAP, merge"));
        assert!(raw.contains("filename=\"audio.wav\""));
    }

    /// Against a real server, so ignored by default. Only send made-up test
    /// clips to a server outside your organisation's control.
    ///   REMOTE_ASR_URL=https://host REMOTE_ASR_CLIP=clip.wav \
    ///     cargo test --lib live_server -- --ignored --nocapture
    /// REMOTE_ASR_KEY_FILE overrides ~/.config/asr/key; REMOTE_ASR_WORDS is a
    /// comma-separated custom word list.
    #[test]
    #[ignore]
    fn live_server_transcribes_a_wav_clip() {
        let url = std::env::var("REMOTE_ASR_URL").expect("set REMOTE_ASR_URL");
        let clip = std::env::var("REMOTE_ASR_CLIP").expect("set REMOTE_ASR_CLIP");
        let reader = hound::WavReader::open(&clip).unwrap();
        assert_eq!(
            reader.spec().sample_rate,
            16000,
            "the recorder produces 16 kHz"
        );
        let samples: Vec<f32> = reader
            .into_samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect();
        let request = RemoteAsrRequest {
            url,
            model: "Qwen/Qwen3-ASR-1.7B".to_string(),
            key_file: std::env::var("REMOTE_ASR_KEY_FILE")
                .unwrap_or_else(|_| "~/.config/asr/key".to_string()),
            custom_words: std::env::var("REMOTE_ASR_WORDS")
                .map(|w| w.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_default(),
            language: "auto".to_string(),
        };
        let started = std::time::Instant::now();
        let text = transcribe(&request, &samples).unwrap();
        println!("{:.2}s: {text}", started.elapsed().as_secs_f64());
        assert!(!text.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_error_status_becomes_a_readable_error_without_the_key() {
        let (base, server) =
            one_shot_server("HTTP/1.1 401 Unauthorized\r\nContent-Length: 12\r\n\r\nUnauthorized")
                .await;
        let mut key_file = tempfile::NamedTempFile::new().unwrap();
        writeln!(key_file, "wrong-key").unwrap();
        let request = request_for(base, &key_file);

        let error = tokio::task::spawn_blocking(move || transcribe(&request, &[0.0; 160]))
            .await
            .unwrap()
            .unwrap_err()
            .to_string();
        server.await.unwrap();
        assert!(error.contains("401"), "{error}");
        assert!(error.contains("key file"), "{error}");
        assert!(!error.contains("wrong-key"), "{error}");
    }
}
