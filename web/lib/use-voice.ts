"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Voice I/O via the browser Web Speech API — no server, no key.
 *  - listen(): SpeechRecognition (mic → text). Chrome/Edge/Safari; feature-detected.
 *  - speak():  SpeechSynthesis (text → spoken). Widely supported.
 * Degrades gracefully: `supported` flags let the UI hide controls the browser can't do.
 */
export function useVoice() {
  const [supported, setSupported] = useState({ stt: false, tts: false });
  const [listening, setListening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recRef = useRef<any>(null);

  useEffect(() => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    setSupported({ stt: !!SR, tts: "speechSynthesis" in window });
    return () => {
      try {
        recRef.current?.abort?.();
        window.speechSynthesis?.cancel?.();
      } catch {
        /* noop */
      }
    };
  }, []);

  const listen = useCallback((onResult: (text: string) => void) => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) return;
    setError(null);
    const rec = new SR();
    rec.lang = "en-US";
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    rec.onresult = (e: any) => {
      const text = e.results?.[0]?.[0]?.transcript ?? "";
      if (text) onResult(text);
    };
    rec.onend = () => setListening(false);
    // Surface the failure (blocked permission is the most common) instead of silently going idle.
    rec.onerror = (e: any) => {
      setListening(false);
      if (e?.error === "not-allowed" || e?.error === "service-not-allowed") {
        setError("Mic access was blocked — allow the microphone in your browser, or just type instead.");
      } else if (e?.error === "no-speech") {
        setError("Didn't catch that — tap the mic and try again, or type instead.");
      } else {
        setError("Voice input failed — please type your request instead.");
      }
    };
    recRef.current = rec;
    setListening(true);
    try {
      rec.start();
    } catch {
      setListening(false);
      setError("Voice input failed to start — please type your request instead.");
    }
  }, []);

  const stopListening = useCallback(() => {
    try {
      recRef.current?.stop();
    } catch {
      /* noop */
    }
    setListening(false);
  }, []);

  const speak = useCallback((text: string) => {
    if (!("speechSynthesis" in window) || !text) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text.replace(/[`*_#>]/g, ""));
    u.rate = 1.03;
    u.pitch = 1.0;
    window.speechSynthesis.speak(u);
  }, []);

  return { supported, listening, error, listen, stopListening, speak };
}
