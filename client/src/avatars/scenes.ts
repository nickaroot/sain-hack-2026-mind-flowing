// Rive-сцены экспертов. На каждого эксперта — одна живая сцена на всё приложение: она рисуется на свой
// невидимый холст, а каждый аватар этого эксперта (в галерее, шапке, ответах, меню) копирует с него свой кадр —
// по пояс или лицо. Так сцена загружается один раз, а графический контекст на странице один, сколько бы ни было аватаров.
// Сцены собираются в rive-lab/ и управляются числами 0–100: взгляд (lookX, lookY) и рот. У сцены с ртом-объектом
// рот — это open (0 — улыбка покоя, 25 — губы сомкнуты, 100 — широко открыт) и round (округление);
// у прежних сцен — одно число mouth.
import type { AvatarState } from "./index"
import type { MouthShape } from "../visemes"
import biohacker from "./biohacker.riv?url"
import clinician from "./clinician.riv?url"
import nutritionist from "./nutritionist.riv?url"

const FILES: Record<string, string> = { clinician, biohacker, nutritionist }
/** Сторона холста сцены в пикселях: хватает большому аватару на экране с двойной плотностью. */
const SIDE = 512

/** Один аватар на экране: куда копировать кадр, какую часть сцены и что эксперт сейчас делает. */
export type Viewer = {
  canvas: HTMLCanvasElement
  /** Во сколько раз кадр больше круга и какая точка кадра в его центре (доли кадра). */
  zoom: number
  focus: { x: number; y: number }
  read: () => { state: AvatarState; getLevel?: () => number | null; getMouth?: () => MouthShape | null }
}
type Status = "loading" | "ready" | "failed"

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))
const RANK: Record<AvatarState, number> = { idle: 0, active: 1, listening: 2, thinking: 3, speaking: 4 }

/** Громкость голоса (0…1) → открытость рта (0–100): те же пороги, что у кадровых аватаров, но без ступенек. */
function openness(level: number): number {
  const stops = [
    [0.06, 0],
    [0.22, 34],
    [0.5, 67],
    [0.8, 100],
  ]
  if (level <= stops[0][0]) return 0
  for (let index = 1; index < stops.length; index++) {
    const [x0, y0] = stops[index - 1]
    const [x1, y1] = stops[index]
    if (level <= x1) return y0 + ((level - x0) / (x1 - x0)) * (y1 - y0)
  }
  return 100
}

const pointer = { x: 0, y: 0, seen: false }
let listening = false
function watchPointer() {
  if (listening) return
  listening = true
  window.addEventListener(
    "pointermove",
    (event) => {
      pointer.x = event.clientX
      pointer.y = event.clientY
      pointer.seen = event.pointerType === "mouse"
    },
    { passive: true },
  )
}

class Scene {
  status: Status = "loading"
  private viewers = new Set<Viewer>()
  private listeners = new Set<() => void>()
  private source = document.createElement("canvas")
  private frame = 0
  private step: ((now: number) => void) | null = null

  constructor(src: string) {
    this.source.width = this.source.height = SIDE
    // Холст сцены стоит в документе за краем экрана: вне документа среда выполнения на него не рисует.
    this.source.setAttribute("aria-hidden", "true")
    this.source.style.cssText = `position:fixed;left:-${SIDE * 2}px;top:0;width:${SIDE}px;height:${SIDE}px;pointer-events:none`
    document.body.appendChild(this.source)
    watchPointer()
    this.load(src).catch(() => this.settle("failed"))
  }

  private settle(status: Status) {
    this.status = status
    this.listeners.forEach((notify) => notify())
  }

  private async load(src: string) {
    // Среда выполнения Rive и её wasm подгружаются отдельным куском и с этого же сервера, не с чужого CDN.
    const [{ Rive, RuntimeLoader, Layout, Fit, EventType }, wasm] = await Promise.all([import("@rive-app/webgl2"), import("@rive-app/webgl2/rive.wasm?url")])
    RuntimeLoader.setWasmUrl(wasm.default)
    const rive = new Rive({
      src,
      canvas: this.source,
      stateMachine: "Avatar",
      autoplay: true,
      autoBind: true,
      shouldDisableRiveListeners: true,
      // Общий графический контекст на все сцены; у холста сцены обычный двумерный, с него можно копировать.
      useOffscreenRenderer: true,
      layout: new Layout({ fit: Fit.Contain }),
      onLoadError: () => this.settle("failed"),
      onLoad: () => {
        const data = rive.viewModelInstance
        const lookX = data?.number("lookX")
        const lookY = data?.number("lookY")
        // Рот с двумя осями: форма под букву. У прежних сцен — одно число, открытие по громкости.
        const shape = { open: data?.number("open"), round: data?.number("round") }
        const shaped = !!(shape.open && shape.round)
        const mouth = shaped ? null : data?.number("mouth")
        if (!lookX || !lookY || !(shaped || mouth)) return this.settle("failed")
        // Аватары показывают сцену, когда на её холсте есть кадр. Событие приходит в начале каждого прохода
        // отрисовки, так что ко второму первый кадр уже нарисован.
        let passes = 0
        const onPass = () => {
          if (++passes < 2) return
          rive.off(EventType.Advance, onPass)
          this.settle("ready")
        }
        rive.on(EventType.Advance, onPass)

        let open = 0
        let round = 0
        let x = 50
        let y = 50
        let last = performance.now()
        this.step = (now: number) => {
          const dt = Math.min(0.05, (now - last) / 1000)
          last = now
          // Сценой управляет тот аватар, у которого сейчас самое «громкое» состояние: говорящий важнее думающего.
          let lead: Viewer | null = null
          let top: ReturnType<Viewer["read"]> = { state: "idle" }
          for (const viewer of this.viewers) {
            const current = viewer.read()
            if (!lead || RANK[current.state] > RANK[top.state]) {
              lead = viewer
              top = current
            }
          }
          const speaking = top.state === "speaking"
          let target = 0
          let pucker = 0
          if (speaking) {
            const said = shaped ? (top.getMouth?.() ?? null) : null
            if (said) {
              // 0 — губы сомкнуты по нарисованной линии, 100 — рот открыт широко; round собирает губы к середине.
              target = said.open
              pucker = said.round
            } else if (shaped) {
              const heard = top.getLevel?.() ?? null
              target = heard !== null ? openness(heard) : Math.max(0, 55 + 45 * Math.sin(now / 90) * Math.sin(now / 310))
            } else {
              const heard = top.getLevel?.() ?? null
              // Без живого звука — правдоподобный ритм слогов.
              target = heard !== null ? openness(heard) : Math.max(0, 55 + 45 * Math.sin(now / 90) * Math.sin(now / 310))
            }
          }
          // Рот догоняет нужную форму быстро, но не скачком: это движение, а не смена кадров.
          // После речи рот закрывается спокойнее, чем двигается в ней.
          const pace = !shaped ? (target > open ? 38 : 16) : !speaking ? 12 : target > open ? 30 : 22
          open += (target - open) * (1 - Math.exp(-dt * pace))
          round += (pucker - round) * (1 - Math.exp(-dt * 16))
          let tx = 50
          let ty = 50
          if (top.state === "thinking") {
            tx = 72 + 8 * Math.sin(now / 1700)
            ty = 18
          } else if (top.state !== "idle" && pointer.seen && lead) {
            const box = lead.canvas.getBoundingClientRect()
            tx = 50 + 50 * clamp((pointer.x - (box.left + box.width / 2)) / 420, -1, 1)
            ty = 50 + 50 * clamp((pointer.y - (box.top + box.height * 0.35)) / 420, -1, 1)
          }
          x += (tx - x) * (1 - Math.exp(-dt * 7))
          y += (ty - y) * (1 - Math.exp(-dt * 7))
          if (shaped) {
            shape.open!.value = open
            shape.round!.value = round
          } else mouth!.value = open
          lookX.value = x
          lookY.value = y

          if (this.status === "ready") {
            for (const viewer of this.viewers) {
              const context = viewer.canvas.getContext("2d")
              if (!context) continue
              const { width, height } = viewer.canvas
              const part = SIDE / viewer.zoom
              context.clearRect(0, 0, width, height)
              context.drawImage(this.source, viewer.focus.x * SIDE - part / 2, viewer.focus.y * SIDE - part / 2, part, part, 0, 0, width, height)
            }
          }
          this.frame = this.viewers.size ? requestAnimationFrame(this.step!) : 0
        }
        if (this.viewers.size) this.frame = requestAnimationFrame(this.step)
      },
    })
  }

  watch(viewer: Viewer, notify: () => void): () => void {
    this.viewers.add(viewer)
    this.listeners.add(notify)
    if (this.step && !this.frame) this.frame = requestAnimationFrame(this.step)
    return () => {
      this.viewers.delete(viewer)
      this.listeners.delete(notify)
    }
  }
}

const scenes = new Map<string, Scene>()

export const hasScene = (expertId: string) => expertId in FILES

/** Подключить аватар к сцене эксперта. Возвращает отписку и способ узнать, готова ли сцена. */
export function watchScene(expertId: string, viewer: Viewer, notify: () => void): { status: () => Status; stop: () => void } {
  let scene = scenes.get(expertId)
  if (!scene) {
    scene = new Scene(FILES[expertId])
    scenes.set(expertId, scene)
  }
  const current = scene
  return { status: () => current.status, stop: current.watch(viewer, notify) }
}
