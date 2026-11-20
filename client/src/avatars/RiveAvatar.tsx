import { useEffect, useRef, useState } from "react"
import type { AvatarProps } from "./index"
import { SpriteAvatar, layer } from "./SpriteAvatar"
import { SPRITES } from "./sprites"
import { hasScene, watchScene } from "./scenes"

/**
 * Аватар эксперта из Rive-сцены: сетка с костями дышит и поворачивает голову, рот идёт за голосом, взгляд — за курсором.
 * Большой показывает персонажа по пояс, маленький — лицо; оба копируют кадр одной и той же сцены (scenes.ts).
 * Пока сцена грузится — кадр покоя из листа; если не загрузилась или сцены у эксперта нет — прежний аватар на кадрах.
 */
export function RiveAvatar(props: AvatarProps) {
  const { expertId, state, size, getLevel } = props
  const set = SPRITES[expertId]
  const large = size >= 96
  const canvas = useRef<HTMLCanvasElement>(null)
  const [status, setStatus] = useState<"loading" | "ready" | "failed">("loading")
  const live = useRef({ state, getLevel })
  live.current = { state, getLevel }
  const zoom = large ? set.zoom : set.faceZoom
  const focus = large ? set.focus : set.face

  useEffect(() => {
    const element = canvas.current
    if (!element || !hasScene(expertId)) return
    const link = watchScene(expertId, { canvas: element, zoom, focus, read: () => live.current }, () => setStatus(link.status()))
    setStatus(link.status())
    return link.stop
    // Точка кадра задана числами из sprites.ts и для эксперта не меняется.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [expertId, zoom])

  if (!hasScene(expertId) || status === "failed") return <SpriteAvatar {...props} />

  const ratio = Math.min(2, window.devicePixelRatio || 1)
  return (
    <span className="avatar-live" data-state={state} data-large={large} style={{ width: size, height: size, background: set.bg, borderRadius: "50%" }} aria-hidden="true">
      {status !== "ready" && <i style={layer(set, set.rest, size, large)} />}
      <canvas ref={canvas} className="avatar-rive" data-ready={status === "ready"} width={Math.round(size * ratio)} height={Math.round(size * ratio)} />
    </span>
  )
}
