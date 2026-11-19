/// <reference types="vite/client" />

/** true в сборке `vite --mode mock`: клиент работает на заглушках. */
declare const __MOCK__: boolean

interface ImportMetaEnv {
  /** Адрес серверной части. По умолчанию — стенд команды. */
  readonly VITE_API_URL?: string
  /** Пользователь набора, от чьего имени идёт разговор (нужен только на наборе разработки). */
  readonly VITE_USER_ID?: string
}
