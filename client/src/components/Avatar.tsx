import { UserRound } from "lucide-react"
import { customAvatars } from "../avatars"
import type { AvatarState } from "../avatars"
import type { Persona } from "../experts"

type Props = { persona: Persona; state: AvatarState; size: number; getLevel?: () => number | null }

/** Аватар эксперта: свой компонент из avatars/, а пока его нет — плейсхолдер с теми же состояниями. */
export function Avatar({ persona, state, size, getLevel }: Props) {
  const Custom = customAvatars[persona.id]
  if (Custom) return <Custom expertId={persona.id} name={persona.name} state={state} size={size} getLevel={getLevel} />

  // Маленький размер — значок роли в плитке; большой — портретный круг-заготовка.
  if (size < 96) {
    const Icon = persona.icon
    return (
      <span className="avatar-tile" data-state={state} style={{ width: size, height: size, borderRadius: "50%" }} aria-hidden="true">
        {Icon ? <Icon size={Math.round(size * 0.56)} strokeWidth={2} /> : <b style={{ fontSize: size * 0.5 }}>{persona.name[0]}</b>}
      </span>
    )
  }
  return (
    <span className="avatar-stub" data-state={state} style={{ width: size, height: size }} aria-hidden="true">
      <UserRound size={Math.round(size * 0.4)} strokeWidth={1.2} />
      <span>место для аватара</span>
    </span>
  )
}
