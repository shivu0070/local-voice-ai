'use client';

import React, { useEffect, useRef, useState, useMemo } from 'react';
import { motion } from 'motion/react';
import { useSessionContext, useSessionMessages } from '@livekit/components-react';
import type { AppConfig } from '@/app-config';
import { PreConnectMessage } from '@/components/app/preconnect-message';
import { TileLayout } from '@/components/app/tile-layout';
import {
  AgentControlBar,
  type ControlBarControls,
} from '@/components/livekit/agent-control-bar/agent-control-bar';
import { cn } from '@/lib/utils';
import { ScrollArea } from '../livekit/scroll-area/scroll-area';
import { RoomEvent } from 'livekit-client';
import { MicrophoneIcon } from '@phosphor-icons/react/dist/ssr';

const MotionBottom = motion.create('div');

const BOTTOM_VIEW_MOTION_PROPS = {
  variants: {
    visible: {
      opacity: 1,
      translateY: '0%',
    },
    hidden: {
      opacity: 0,
      translateY: '100%',
    },
  },
  initial: 'hidden',
  animate: 'visible',
  exit: 'hidden',
  transition: {
    duration: 0.3,
    delay: 0.5,
    ease: 'easeOut',
  },
};

interface FadeProps {
  top?: boolean;
  bottom?: boolean;
  className?: string;
}

export function Fade({ top = false, bottom = false, className }: FadeProps) {
  return (
    <div
      className={cn(
        'from-background pointer-events-none h-4 bg-linear-to-b to-transparent',
        top && 'bg-linear-to-b',
        bottom && 'bg-linear-to-t',
        className
      )}
    />
  );
}

interface SessionViewProps {
  appConfig: AppConfig;
}

export const SessionView = ({
  appConfig,
  ...props
}: React.ComponentProps<'section'> & SessionViewProps) => {
  const session = useSessionContext();
  const { messages } = useSessionMessages(session);
  // Always show transcript (ChatGPT-like log), no text input.
  const chatOpen = true;
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const [dataMessages, setDataMessages] = useState<
    { id: string; timestamp: number; message: string; role?: string }[]
  >([]);

  // Listen to raw data packets (lk.chat / lk-chat-topic) as a fallback for chat log
  useEffect(() => {
    const room = session.room;
    if (!room) return;
    const handler = (payload: Uint8Array, participant: any, _?: any, topic?: string) => {
      if (topic !== 'lk.chat' && topic !== 'lk-chat-topic') return;
      try {
        const txt = new TextDecoder().decode(payload);
        const obj = JSON.parse(txt);
        if (obj && obj.message) {
          setDataMessages((prev) => [
            ...prev,
            {
              id: obj.id || `${Date.now()}-${prev.length}`,
              timestamp: obj.timestamp || Date.now(),
              message: obj.message,
              role: obj.role,
            },
          ]);
        }
      } catch {
        /* ignore */
      }
    };
    room.on(RoomEvent.DataReceived, handler);
    return () => {
      room.off(RoomEvent.DataReceived, handler);
    };
  }, [session.room]);

  const displayMessages = useMemo(() => {
    type Normalized = { id: string; timestamp: number; message: string; role: 'user' | 'assistant' };
    const normalizeLivekit = messages.map<Normalized>((m, idx) => ({
      id: m.id ?? `${m.timestamp}-${idx}`,
      timestamp: m.timestamp ?? Date.now(),
      message: m.message ?? '',
      role: m.from?.isLocal ? 'user' : 'assistant',
    }));
    const normalizeData = dataMessages.map<Normalized>((m, idx) => ({
      id: m.id ?? `data-${m.timestamp}-${idx}`,
      timestamp: m.timestamp ?? Date.now(),
      message: m.message ?? '',
      role: (m.role as 'user' | 'assistant') ?? 'assistant',
    }));

    const combined = [...normalizeLivekit, ...normalizeData];

    // collapse updates that reuse the same id (keep newest payload)
    const indexById = new Map<string, number>();
    const ordered: Normalized[] = [];
    combined.forEach((m) => {
      if (indexById.has(m.id)) {
        const pos = indexById.get(m.id)!;
        ordered[pos] = m;
        return;
      }
      indexById.set(m.id, ordered.length);
      ordered.push(m);
    });

    // drop consecutive duplicates (same role + identical text)
    const deduped = ordered.filter((m, i) => {
      const prev = ordered[i - 1];
      if (!prev) return true;
      return !(prev.role === m.role && prev.message.trim() === m.message.trim());
    });

    // Collapse to one user + one assistant per turn.
    const turns: Normalized[] = [];
    let currentUser: Normalized | null = null;
    let currentAssistant: Normalized | null = null;

    const flushTurn = () => {
      if (currentUser) turns.push(currentUser);
      if (currentAssistant) turns.push(currentAssistant);
      currentUser = null;
      currentAssistant = null;
    };

    for (const m of deduped) {
      if (m.role === 'user') {
        flushTurn();
        currentUser = m; // keep last user partial for this turn
      } else {
        // assistant message belongs to current turn; keep only the latest text
        currentAssistant = m;
      }
    }
    flushTurn();

    return turns;
  }, [messages, dataMessages]);

  const controls: ControlBarControls = {
    leave: true,
    microphone: true,
    chat: false, // voice-only UX
    camera: false,
    screenShare: false,
  };

  useEffect(() => {
    const lastMessage = messages.at(-1);
    const lastMessageIsLocal = lastMessage?.from?.isLocal === true;

    if (scrollAreaRef.current && lastMessageIsLocal) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  const previewText = (text: string, words = 5) => {
    const parts = text.trim().split(/\s+/);
    if (parts.length <= words) return text.trim();
    return `${parts.slice(0, words).join(' ')}...`;
  };

  const formatTime = (ts: number) =>
    new Date(ts).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });

  return (
    <section className="bg-background relative z-10 h-full w-full overflow-hidden" {...props}>
      {/* Voice Chat Log */}
      <div className="fixed inset-x-0 top-8 bottom-32 z-30 flex justify-center px-3 md:px-8">
        <div className="bg-background/90 border border-input/40 backdrop-blur-md shadow-2xl shadow-black/30 rounded-3xl w-full max-w-5xl">
          <div className="flex items-center gap-2 px-4 py-3 text-sm text-muted-foreground">
            <MicrophoneIcon weight="bold" /> Voice conversation log
          </div>
          <div className="h-px w-full bg-border" />
          <ScrollArea ref={scrollAreaRef} className="h-[calc(70vh-64px)] px-5 py-4">
            {displayMessages.length === 0 ? (
              <p className="text-muted-foreground text-sm">Start talking to see the conversation.</p>
            ) : (
              <div className="space-y-4">
                {displayMessages.map((m) =>
                  m.role === 'user' ? (
                    <div key={m.id} className="flex justify-end">
                      <div className="max-w-[65%] text-right space-y-1">
                        <div className="inline-flex rounded-full bg-primary px-4 py-2 text-sm italic text-primary-foreground shadow-md">
                          "Listening..."
                        </div>
                        <div className="flex items-center justify-end gap-1 text-[11px] text-muted-foreground">
                          <MicrophoneIcon className="h-4 w-4" weight="fill" />
                          {formatTime(m.timestamp)}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div key={m.id} className="flex justify-start">
                      <div className="max-w-[78%] rounded-2xl bg-muted/80 px-4 py-3 text-base leading-6 text-foreground shadow-inner">
                        <div className="whitespace-normal break-words">{m.message}</div>
                        <div className="mt-1 text-[11px] text-muted-foreground">{formatTime(m.timestamp)}</div>
                      </div>
                    </div>
                  )
                )}
              </div>
            )}
          </ScrollArea>
        </div>
      </div>

      {/* Tile Layout */}
      <TileLayout chatOpen={chatOpen} />

      {/* Bottom */}
      <MotionBottom
        {...BOTTOM_VIEW_MOTION_PROPS}
        className="fixed inset-x-3 bottom-0 z-50 md:inset-x-12"
      >
        {appConfig.isPreConnectBufferEnabled && (
          <PreConnectMessage messages={messages} className="pb-4" />
        )}
        <div className="bg-background relative mx-auto max-w-2xl pb-3 md:pb-12">
          <Fade bottom className="absolute inset-x-0 top-0 h-4 -translate-y-full" />
          <AgentControlBar
            controls={controls}
            isConnected={session.isConnected}
            onDisconnect={session.end}
          />
        </div>
      </MotionBottom>
    </section>
  );
};
