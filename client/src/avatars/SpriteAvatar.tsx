import { useEffect, useRef, useState } from "react"
import type { CSSProperties } from "react"
import type { AvatarProps, AvatarState } from "./index"
import { COLS, ROWS, SPRITES } from "./sprites"
import type { SpriteSet } from "./sprites"

type Pose = { cur: number; prev: number; seq: number; pop: number }

const between = (min: number, max: number) => min + Math.random() * (max - min)

/**
 * Режиссёр мимики: по состоянию аватара решает, какой кадр показать и когда.
 * Плавность дают три слоя: кадры сменяются коротким наплывом, корпус непрерывно «дышит» (CSS),
 * а при выборе эксперта проигрывается приветствие с пружинящим «прыжком».
 */
function usePose(set: SpriteSet, state: AvatarState, lively: boolean, getLevel?: () => number | null): Pose {
  const [pose, setPose] = useState<Pose>({ cur: set.rest, prev: set.rest, seq: 0, pop: 0 })
  const before = useRef<AvatarState>("idle")
  const level = useRef(getLevel)
  level.current = getLevel

  useEffect(() => {
    let alive = true
    let timer = 0
    const show = (frame: number) => setPose((pose) => (pose.cur === frame ? pose : { ...pose, cur: frame, prev: pose.cur, seq: pose.seq + 1 }))
    const later = (ms: number, next: () => void) => {
      timer = window.setTimeout(() => alive && next(), ms)
    }
    const wasIdle = before.current === "idle"
    before.current = state

    // Моргание и редкая улыбка — пока эксперт выбран и молчит.
    const rest = (frame: number) => {
      show(frame)
      later(between(2200, 4600), () => {
        if (Math.random() < 0.25) {
          show(set.smile)
          later(900, () => rest(frame))
        } else {
          show(set.blink)
          later(130, () => rest(frame))
        }
      })
    }

    // Речь: при живом звуке рот идёт за громкостью голоса, иначе — правдоподобный ритм слогов.
    let open = 0
    const talk = () => {
      const heard = level.current?.() ?? null
      if (heard !== null) {
        open = heard < 0.06 ? 0 : heard < 0.22 ? 1 : heard < 0.5 ? 2 : 3
      } else {
        const pause = Math.random() < 0.16
        open = pause ? 0 : Math.max(1, Math.min(3, open + (Math.random() < 0.5 ? 1 : -1) + (open === 0 ? 1 : 0)))
      }
      const round = set.round !== null && open === 2 && Math.random() < 0.12
      show(round ? set.round! : set.talk[open])
      later(heard !== null ? 80 : between(95, 150), talk)
    }

    const think = (step: number) => {
      show(set.think[step % set.think.length])
      later(between(1300, 2300), () => {
        show(set.blink)
        later(130, () => think(step + 1))
      })
    }

    const run = () => {
      if (state === "speaking") talk()
      else if (state === "thinking") think(0)
      else if (state === "listening") rest(set.listen)
      else rest(set.rest)
    }

    if (state === "idle" || (!lively && state === "active")) {
      show(set.rest)
    } else if (lively && wasIdle) {
      // Эксперта только что выбрали: приветствие и «прыжок».
      setPose((pose) => ({ ...pose, pop: pose.pop + 1 }))
      const greet = (step: number) => {
        if (step >= set.greet.length) return run()
        show(set.greet[step])
        later(120, () => greet(step + 1))
      }
      greet(0)
    } else {
      run()
    }
    return () => {
      alive = false
      clearTimeout(timer)
    }
  }, [set, state, lively])

  return pose
}

export const layer = (set: SpriteSet, frame: number, size: number, large: boolean): CSSProperties => {
  const zoom = large ? set.zoom : set.faceZoom
  const focus = large ? set.focus : set.face
  const side = zoom * size
  return {
    width: side,
    height: side,
    left: size / 2 - focus.x * side,
    top: size / 2 - focus.y * side,
    backgroundImage: `url(${set.src})`,
    backgroundSize: `${COLS * 100}% ${ROWS * 100}%`,
    backgroundPosition: `${((frame % COLS) / (COLS - 1)) * 100}% ${(Math.floor(frame / COLS) / (ROWS - 1)) * 100}%`,
  }
}

/** Аватар-персонаж из листа кадров. Большой — в круге по пояс, маленький — значок с лицом. */
export function SpriteAvatar({ expertId, state, size, getLevel }: AvatarProps) {
  const set = SPRITES[expertId]
  const large = size >= 96
  const pose = usePose(set, state, large, getLevel)
  return (
    <span
      className="avatar-live"
      data-state={state}
      data-large={large}
      style={{ width: size, height: size, background: set.bg, borderRadius: "50%" }}
      aria-hidden="true"
    >
      <span className="avatar-pop" key={pose.pop}>
        <span className="avatar-motion">
          <i style={layer(set, pose.prev, size, large)} />
          <i key={pose.seq} className="avatar-fade" style={layer(set, pose.cur, size, large)} />
        </span>
      </span>
    </span>
  )
}
