import { Button } from '@/components/livekit/button';
import { SpinnerIcon, MicrophoneIcon } from '@phosphor-icons/react/dist/ssr';
import { useState } from 'react';

interface WelcomeViewProps {
  startButtonText: string;
  onStartCall: () => void;
}

export const WelcomeView = ({
  startButtonText,
  onStartCall,
  ref,
}: React.ComponentProps<'div'> & WelcomeViewProps) => {
  const [connecting, setConnecting] = useState(false);

  const handleStart = () => {
    setConnecting(true);
    onStartCall();
  };

  return (
    <div ref={ref}>
      <section className="bg-background flex flex-col items-center justify-center text-center gap-4">
        <div className="bg-primary/10 text-primary flex h-14 w-14 items-center justify-center rounded-full">
          <MicrophoneIcon weight="fill" size={26} />
        </div>

        <p className="text-foreground max-w-prose leading-6 font-medium">
          Tap to connect and start talking.
        </p>

        <Button
          variant="primary"
          size="lg"
          onClick={handleStart}
          disabled={connecting}
          className="mt-1 w-48 font-medium"
        >
          {connecting ? (
            <span className="flex items-center gap-2 justify-center">
              <SpinnerIcon className="animate-spin" weight="bold" /> Connecting…
            </span>
          ) : (
            startButtonText
          )}
        </Button>
      </section>

      <div className="fixed bottom-5 left-0 flex w-full items-center justify-center">
        <p className="text-muted-foreground max-w-prose pt-1 text-xs leading-5 font-normal text-pretty md:text-sm">
          Need help getting set up? Ask the team or check your project docs.
        </p>
      </div>
    </div>
  );
};
