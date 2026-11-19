import { useEffect, useRef, useState } from "react"
import type { ReactNode } from "react"
import { X } from "lucide-react"

/** Держит элемент в дереве, пока играет анимация ухода. */
export function usePresence(open: boolean, exitMs = 260): { mounted: boolean; visible: boolean } {
  const [mounted, setMounted] = useState(open)
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    if (open) {
      setMounted(true)
      const frame = requestAnimationFrame(() => requestAnimationFrame(() => setVisible(true)))
      return () => cancelAnimationFrame(frame)
    }
    setVisible(false)
    const timer = setTimeout(() => setMounted(false), exitMs)
    return () => clearTimeout(timer)
  }, [open, exitMs])
  return { mounted, visible }
}

type Props = {
  open: boolean
  onClose: () => void
  title: string
  /** sheet — лист снизу на телефоне и окно по центру на широком экране; menu — на широком экране меню под шапкой. */
  kind?: "sheet" | "menu"
  children: ReactNode
}

export function Sheet({ open, onClose, title, kind = "sheet", children }: Props) {
  const { mounted, visible } = usePresence(open)
  const panel = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const previous = document.activeElement as HTMLElement | null
    panel.current?.focus()
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onClose()
    document.addEventListener("keydown", onKey)
    return () => {
      document.removeEventListener("keydown", onKey)
      previous?.focus?.()
    }
  }, [open, onClose])

  if (!mounted) return null
  return (
    <div className="overlay" data-kind={kind} data-open={visible}>
      <div className="overlay-backdrop" onClick={onClose} aria-hidden="true" />
      <div className="sheet" role="dialog" aria-modal="true" aria-label={title} tabIndex={-1} ref={panel}>
        <span className="sheet-handle" aria-hidden="true" />
        <div className="sheet-head">
          <h2>{title}</h2>
          <button type="button" className="icon-button" aria-label="Закрыть" onClick={onClose}>
            <X size={20} />
          </button>
        </div>
        <div className="sheet-body">{children}</div>
      </div>
    </div>
  )
}
