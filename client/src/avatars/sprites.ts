// Аватары-персонажи: у каждого лист из 12 кадров-выражений (4×3, кадр 480×480, края растворены в прозрачность).
// Кадры собраны из исходных роликов и выровнены по корпусу, поэтому смена кадра читается как смена мимики.
// Номера кадров — с нуля, слева направо, сверху вниз.
import biohacker from "./biohacker.webp"
import clinician from "./clinician.webp"
import nutritionist from "./nutritionist.webp"

export const COLS = 4
export const ROWS = 3

export type SpriteSet = {
  src: string
  /** Цвет фона исходных кадров: им залит круг, чтобы края кадра не были видны. */
  bg: string
  rest: number
  smile: number
  blink: number
  /** Рот от закрытого к широко открытому: четыре ступени для речи. */
  talk: [number, number, number, number]
  /** Рот «о», если такой кадр есть. */
  round: number | null
  /** Задумчивые кадры: взгляд в сторону, прищур. */
  think: number[]
  listen: number
  /** Приветствие при выборе эксперта: кадры подряд. */
  greet: number[]
  /** Большой аватар: во сколько раз кадр больше круга и какая точка кадра в центре (доли кадра). */
  zoom: number
  focus: { x: number; y: number }
  /** Маленький значок: крупно лицо. */
  faceZoom: number
  face: { x: number; y: number }
}

export const SPRITES: Record<string, SpriteSet> = {
  clinician: {
    src: clinician,
    bg: "#F4CFB5",
    rest: 11,
    smile: 1,
    blink: 10,
    talk: [11, 1, 2, 5],
    round: 3,
    think: [7, 9],
    listen: 0,
    greet: [2, 5, 5, 2],
    zoom: 1.2,
    focus: { x: 0.5, y: 0.41 },
    faceZoom: 2.3,
    face: { x: 0.5, y: 0.31 },
  },
  biohacker: {
    src: biohacker,
    bg: "#EFCDC1",
    rest: 9,
    smile: 0,
    blink: 10,
    talk: [9, 8, 2, 5],
    round: 3,
    think: [7, 1],
    listen: 0,
    greet: [2, 5, 6, 5],
    zoom: 1.06,
    focus: { x: 0.5, y: 0.42 },
    faceZoom: 2.1,
    face: { x: 0.51, y: 0.33 },
  },
  nutritionist: {
    src: nutritionist,
    bg: "#F4D1B8",
    rest: 0,
    smile: 11,
    blink: 10,
    talk: [0, 1, 2, 2],
    round: null,
    think: [7, 9],
    listen: 8,
    greet: [1, 2, 3, 4, 5, 6],
    zoom: 1,
    focus: { x: 0.5, y: 0.43 },
    faceZoom: 2.1,
    face: { x: 0.5, y: 0.3 },
  },
}
