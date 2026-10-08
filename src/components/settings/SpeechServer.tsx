import React from "react";
import { useTranslation } from "react-i18next";
import { SettingContainer } from "../ui/SettingContainer";
import { Input } from "../ui/Input";
import { useSettings } from "../../hooks/useSettings";

interface SpeechServerProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

type SpeechServerSettingKey =
  | "remote_asr_url"
  | "remote_asr_model"
  | "remote_asr_key_file";

const FIELDS: {
  key: SpeechServerSettingKey;
  i18nKey: "url" | "model" | "keyFile";
  placeholder: string;
}[] = [
  {
    key: "remote_asr_url",
    i18nKey: "url",
    placeholder: "https://asr.example.com",
  },
  {
    key: "remote_asr_model",
    i18nKey: "model",
    placeholder: "Qwen/Qwen3-ASR-1.7B",
  },
  {
    key: "remote_asr_key_file",
    i18nKey: "keyFile",
    placeholder: "~/.config/asr/key",
  },
];

/** Settings for the "Speech Server" model, which sends recordings to a server. */
export const SpeechServerSettings: React.FC<SpeechServerProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    return (
      <>
        {FIELDS.map(({ key, i18nKey, placeholder }) => (
          <SettingContainer
            key={key}
            title={t(`settings.advanced.speechServer.${i18nKey}.title`)}
            description={t(
              `settings.advanced.speechServer.${i18nKey}.description`,
            )}
            descriptionMode={descriptionMode}
            grouped={grouped}
          >
            <Input
              type="text"
              value={getSetting(key) ?? ""}
              onChange={(e) => updateSetting(key, e.target.value)}
              placeholder={placeholder}
              disabled={isUpdating(key)}
              spellCheck={false}
              autoCapitalize="off"
              autoCorrect="off"
            />
          </SettingContainer>
        ))}
      </>
    );
  },
);
