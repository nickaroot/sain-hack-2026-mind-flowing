import { Check } from "lucide-react"
import { Avatar } from "./Avatar"
import { Sheet } from "./Sheet"
import { accentStyle } from "../experts"
import type { Persona } from "../experts"

type PickerProps = { open: boolean; personas: Persona[]; activeId: string; onPick: (id: string) => void; onClose: () => void }

/** Смена эксперта посреди разговора: то же место, где в чат-ассистентах выбирают модель. */
export function ExpertPicker({ open, personas, activeId, onPick, onClose }: PickerProps) {
  return (
    <Sheet open={open} onClose={onClose} title="Кто отвечает" kind="menu">
      <p className="sheet-lead">Ответы на уже заданные вопросы готовы у всех — разговор сразу зазвучит другим голосом.</p>
      <div className="picker" role="radiogroup" aria-label="Эксперт">
        {personas.map((persona) => (
          <button
            key={persona.id}
            type="button"
            role="radio"
            aria-checked={persona.id === activeId}
            className="picker-row"
            style={accentStyle(persona.hue)}
            onClick={() => {
              onPick(persona.id)
              onClose()
            }}
          >
            <Avatar persona={persona} size={44} state="active" />
            <span className="picker-text">
              <strong>{persona.name}</strong>
              <span>{persona.tagline}</span>
            </span>
            {persona.id === activeId && <Check size={22} strokeWidth={2.4} className="picker-check" aria-label="Выбран" />}
          </button>
        ))}
      </div>
    </Sheet>
  )
}

/** Полные описания с сервера: они длинные и написаны для нейросети, поэтому спрятаны за «Подробнее». */
export function ExpertAbout({ open, persona, onClose }: { open: boolean; persona: Persona; onClose: () => void }) {
  return (
    <Sheet open={open} onClose={onClose} title={persona.name}>
      <div className="about">
        <h3>Как думает</h3>
        <p>{persona.about}</p>
        <h3>Как говорит</h3>
        <p>{persona.speechStyle}</p>
        <p className="note">Эксперт вымышленный: пересказывает исследования из базы и не ставит диагнозы.</p>
      </div>
    </Sheet>
  )
}
