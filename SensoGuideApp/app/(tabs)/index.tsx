import React, { useState, useEffect, useRef, JSX } from "react";
import {
  View, Text, TouchableOpacity, StyleSheet,
  StatusBar, Animated, Easing,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { Audio } from "expo-av";
import * as Speech from "expo-speech";
import * as Haptics from "expo-haptics";

// ─────────────────────────────────────────────
// CONFIG  ← change MAC_IP to your Mac's IP
// ─────────────────────────────────────────────
const USE_MOCK  = false;
const MAC_IP    = "10.1.150.212";          // ← replace this
const API_BASE  = `http://${MAC_IP}:8000`;

// ─────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────
type Mode = "idle" | "detection" | "text" | "ask" | "listening" | "processing";

// ─────────────────────────────────────────────
// MOCK HELPERS (USE_MOCK = true for offline testing)
// ─────────────────────────────────────────────
const delay = (ms: number) => new Promise(r => setTimeout(r, ms));

const MOCK_COMMANDS = ["detection", "text", "ask", "stop"];
const MOCK_DETECTIONS = [
  "I see 2 persons in front of you.",
  "I see 1 chair and 1 table in front of you.",
  "Clear path.",
];
const MOCK_OCR = [
  "You are looking at a food packaging.",
  "This appears to be a street sign.",
  "I cannot see any text here.",
];
const MOCK_ANSWERS = [
  "There are 2 people in front of you right now.",
  "The text says it is a pharmacy.",
  "I am not sure based on what I can see.",
];

// ─────────────────────────────────────────────
// API CALLS
// ─────────────────────────────────────────────
async function apiSTT(uri: string): Promise<string> {
  if (USE_MOCK) { await delay(800); return MOCK_COMMANDS[Math.floor(Math.random()*4)]; }
  const fd = new FormData();
  fd.append("audio", { uri, type: "audio/m4a", name: "cmd.m4a" } as any);
  const r = await fetch(`${API_BASE}/stt`, { method: "POST", body: fd });
  const j = await r.json();
  return j.transcript || "";
}

async function apiDetect(uri: string): Promise<string> {
  if (USE_MOCK) { await delay(900); return MOCK_DETECTIONS[Math.floor(Math.random()*3)]; }
  const fd = new FormData();
  fd.append("image", { uri, type: "image/jpeg", name: "frame.jpg" } as any);
  const r = await fetch(`${API_BASE}/detect`, { method: "POST", body: fd });
  const j = await r.json();
  return j.description || "Detection failed.";
}

async function apiOCR(uri: string): Promise<string> {
  if (USE_MOCK) { await delay(1000); return MOCK_OCR[Math.floor(Math.random()*3)]; }
  const fd = new FormData();
  fd.append("image", { uri, type: "image/jpeg", name: "frame.jpg" } as any);
  const r = await fetch(`${API_BASE}/ocr`, { method: "POST", body: fd });
  const j = await r.json();
  return j.summary || "Text reading failed.";
}

async function apiAsk(audioUri: string, imageUri: string): Promise<{question:string, answer:string}> {
  if (USE_MOCK) {
    await delay(1500);
    return { question: "What is in front of me?", answer: MOCK_ANSWERS[Math.floor(Math.random()*3)] };
  }
  const fd = new FormData();
  fd.append("audio", { uri: audioUri, type: "audio/m4a", name: "question.m4a" } as any);
  fd.append("image", { uri: imageUri, type: "image/jpeg", name: "frame.jpg" } as any);
  const r = await fetch(`${API_BASE}/ask`, { method: "POST", body: fd });
  const j = await r.json();
  return { question: j.question || "", answer: j.answer || "I could not get an answer." };
}

// ─────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────
const MODE_COLOR: Record<Mode, string> = {
  idle:       "#64748b",
  detection:  "#00d4ff",
  text:       "#a855f7",
  ask:        "#f59e0b",
  listening:  "#22c55e",
  processing: "#f97316",
};

const MODE_LABEL: Record<Mode, string> = {
  idle:       "Hold to speak a command",
  detection:  "Detection active",
  text:       "Reading text",
  ask:        "Ask mode",
  listening:  "Listening…",
  processing: "Processing…",
};

// ─────────────────────────────────────────────
// APP
// ─────────────────────────────────────────────
export default function App(): JSX.Element {
  const [mode,       setMode]       = useState<Mode>("idle");
  const [statusText, setStatusText] = useState("Hold the button and speak a command");
  const [lastResult, setLastResult] = useState("");
  const [audioReady, setAudioReady] = useState(false);
  const [error,      setError]      = useState("");

  const [camPerm, requestCamPerm] = useCameraPermissions();
  const cameraRef    = useRef<any>(null);
  const recordingRef = useRef<Audio.Recording | null>(null);
  const intervalRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const pulseAnim    = useRef(new Animated.Value(1)).current;
  const modeRef      = useRef<Mode>("idle");

  // keep ref in sync for interval callbacks
  useEffect(() => { modeRef.current = mode; }, [mode]);

  // ── Permissions + Audio ──
  useEffect(() => {
    (async () => {
      if (!camPerm?.granted) await requestCamPerm();
      const { granted } = await Audio.requestPermissionsAsync();
      if (!granted) { setError("Microphone permission denied"); return; }
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
        playThroughEarpieceAndroid: false,
      });
      setAudioReady(true);
    })();
    return () => stopAll();
  }, []);

  // ── Pulse animation while listening ──
  useEffect(() => {
    if (mode === "listening") {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.25, duration: 600, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1,    duration: 600, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
        ])
      ).start();
    } else {
      pulseAnim.setValue(1);
    }
  }, [mode]);

  // ── Speak helper ──
  const speak = (text: string) => {
    Speech.stop();
    Speech.speak(text, { rate: 0.85, language: "en-US" });
    setLastResult(text);
  };

  // ── Capture frame from camera ──
  const captureFrame = async (): Promise<string | null> => {
    if (!cameraRef.current) return null;
    try {
      const p = await cameraRef.current.takePictureAsync({ quality: 0.55, skipProcessing: true });
      return p.uri;
    } catch { return null; }
  };

  // ── START recording (button press in) ──
  const onPressIn = async () => {
    if (!audioReady) { setStatusText("Audio not ready, wait a moment"); return; }
    setError("");
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    setMode("listening");
    setStatusText('Say "detection", "text", "ask", or "stop"');

    try {
      const rec = new Audio.Recording();
      await rec.prepareToRecordAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
      await rec.startAsync();
      recordingRef.current = rec;
    } catch (e: any) {
      setError(`Mic error: ${e.message}`);
      setMode("idle");
    }
  };

  // ── STOP recording (button release) ──
  const onPressOut = async () => {
    if (!recordingRef.current) return;

    setMode("processing");
    setStatusText("Understanding command…");

    try {
      await recordingRef.current.stopAndUnloadAsync();
      const uri = recordingRef.current.getURI();
      recordingRef.current = null;
      if (!uri) throw new Error("No audio URI");

      const transcript = await apiSTT(uri);
      console.log("[STT]", transcript);
      handleCommand(transcript.toLowerCase().trim(), uri);
    } catch (e: any) {
      setError(e.message);
      setMode("idle");
      setStatusText("Could not process — try again");
    }
  };

  // ── COMMAND ROUTER ──
  const handleCommand = (cmd: string, audioUri: string) => {
    if (cmd.includes("detect") || cmd.includes("see") || cmd.includes("what"))
      activateDetection();
    else if (cmd.includes("text") || cmd.includes("read") || cmd.includes("lire"))
      activateText();
    else if (cmd.includes("ask") || cmd.includes("question") || cmd.includes("demande"))
      activateAsk(audioUri);
    else if (cmd.includes("stop") || cmd.includes("arrête") || cmd.includes("quit"))
      stopAll();
    else {
      setMode("idle");
      setStatusText(`Didn't understand: "${cmd}"`);
      speak("I did not understand. Say detection, text, ask, or stop.");
    }
  };

  // ── DETECTION MODE — runs every 4s until stopped ──
  const activateDetection = () => {
    stopInterval();
    setMode("detection");
    setStatusText("Scanning…");
    speak("Detection mode activated.");

    const run = async () => {
      if (modeRef.current !== "detection") return;
      const uri = await captureFrame();
      if (!uri) return;
      const desc = await apiDetect(uri);
      setStatusText(desc);
      speak(desc);
    };

    run();   // immediate first run
    intervalRef.current = setInterval(run, 4000);
  };

  // ── TEXT MODE — runs every 6s until stopped ──
  const activateText = () => {
    stopInterval();
    setMode("text");
    setStatusText("Reading text…");
    speak("Text mode activated.");

    const run = async () => {
      if (modeRef.current !== "text") return;
      const uri = await captureFrame();
      if (!uri) return;
      const summary = await apiOCR(uri);
      setStatusText(summary);
      speak(summary);
    };

    run();
    intervalRef.current = setInterval(run, 6000);
  };

  // ── ASK MODE — one shot: record question, capture frame, send both ──
  const activateAsk = async (commandAudioUri: string) => {
    stopInterval();
    setMode("ask");

    // The command audio already contains the question in many cases.
    // But we re-record a fresh dedicated question for accuracy.
    speak("I am listening. Ask your question now.");
    setStatusText("Recording your question…");

    // Wait for speak to start, then record fresh question
    await delay(1000);

    let questionAudioUri = commandAudioUri;  // fallback to command audio

    if (audioReady) {
      try {
        const rec = new Audio.Recording();
        await rec.prepareToRecordAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
        await rec.startAsync();
        await delay(5000);   // 5 seconds to ask
        await rec.stopAndUnloadAsync();
        questionAudioUri = rec.getURI() || commandAudioUri;
      } catch (e: any) {
        console.warn("[Ask recording]", e.message);
        // fall through with command audio
      }
    }

    setMode("processing");
    setStatusText("Thinking…");

    try {
      const imageUri = await captureFrame();
      if (!imageUri) {
        speak("Could not capture camera frame."); setMode("idle"); return;
      }

      const { question, answer } = await apiAsk(questionAudioUri, imageUri);
      console.log("[Ask] Q:", question, "A:", answer);
      setStatusText(answer);
      speak(answer);
    } catch (e: any) {
      setError(e.message);
      speak("Ask mode failed.");
    } finally {
      setMode("idle");
    }
  };

  // ── STOP everything ──
  const stopInterval = () => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
  };

  const stopAll = () => {
    stopInterval();
    Speech.stop();
    if (recordingRef.current) {
      recordingRef.current.stopAndUnloadAsync().catch(() => {});
      recordingRef.current = null;
    }
    setMode("idle");
    setStatusText("Hold the button and speak a command");
    setLastResult("");
  };

  const accent = MODE_COLOR[mode];

  return (
    <View style={s.root}>
      <StatusBar barStyle="light-content" />

      {/* Camera */}
      {camPerm?.granted
        ? <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" />
        : <View style={s.noCam}><Text style={s.white}>Camera permission needed</Text></View>
      }

      {/* Dark overlay */}
      <View style={s.overlay} />

      {/* Mode badge */}
      <View style={[s.badge, { backgroundColor: accent + "33", borderColor: accent }]}>
        <Text style={[s.badgeText, { color: accent }]}>{MODE_LABEL[mode].toUpperCase()}</Text>
      </View>

      {/* Status text */}
      <View style={s.statusBox}>
        <Text style={s.statusText}>{statusText}</Text>
        {!!lastResult && lastResult !== statusText && (
          <Text style={s.resultText} numberOfLines={4}>{lastResult}</Text>
        )}
        {!!error && <Text style={s.errorText}>{error}</Text>}
      </View>

      {/* Main button */}
      <View style={s.btnArea}>
        <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
          <TouchableOpacity
            onPressIn={onPressIn}
            onPressOut={onPressOut}
            activeOpacity={0.85}
            style={[s.btn, { borderColor: accent, shadowColor: accent }]}
          >
            <Text style={[s.btnIcon, { color: accent }]}>
              {mode === "listening" ? "🎙" : mode === "processing" ? "⏳" : "●"}
            </Text>
          </TouchableOpacity>
        </Animated.View>
        <Text style={[s.hint, { color: accent }]}>
          {mode === "listening" ? "Release to send" : "Hold to command"}
        </Text>
      </View>

      {/* Stop button — always visible when active */}
      {mode !== "idle" && (
        <TouchableOpacity onPress={stopAll} style={s.stopBtn}>
          <Text style={s.stopText}>■  STOP</Text>
        </TouchableOpacity>
      )}
    </View>
  );
}

// ─────────────────────────────────────────────
// STYLES
// ─────────────────────────────────────────────
const s = StyleSheet.create({
  root:       { flex: 1, backgroundColor: "#000" },
  overlay:    { ...StyleSheet.absoluteFillObject, backgroundColor: "#000a" },
  noCam:      { flex: 1, justifyContent: "center", alignItems: "center" },
  white:      { color: "#fff" },

  badge: {
    position: "absolute", top: 56, alignSelf: "center",
    paddingHorizontal: 18, paddingVertical: 6,
    borderRadius: 20, borderWidth: 1,
  },
  badgeText:  { fontSize: 12, fontWeight: "700", letterSpacing: 1.5 },

  statusBox: {
    position: "absolute", bottom: 220,
    left: 24, right: 24, alignItems: "center",
  },
  statusText: { color: "#fff", fontSize: 17, textAlign: "center", fontWeight: "600" },
  resultText: { color: "#94a3b8", fontSize: 14, textAlign: "center", marginTop: 8 },
  errorText:  { color: "#ef4444", fontSize: 13, textAlign: "center", marginTop: 6 },

  btnArea: { position: "absolute", bottom: 80, left: 0, right: 0, alignItems: "center" },
  btn: {
    width: 110, height: 110, borderRadius: 55,
    borderWidth: 2.5,
    justifyContent: "center", alignItems: "center",
    shadowOpacity: 0.6, shadowRadius: 20, shadowOffset: { width: 0, height: 0 },
  },
  btnIcon:    { fontSize: 38 },
  hint:       { marginTop: 12, fontSize: 13, fontWeight: "500" },

  stopBtn: {
    position: "absolute", bottom: 28, alignSelf: "center",
    paddingHorizontal: 28, paddingVertical: 10,
    backgroundColor: "#ef444422", borderRadius: 20, borderWidth: 1, borderColor: "#ef4444",
  },
  stopText: { color: "#ef4444", fontWeight: "700", fontSize: 14 },
});