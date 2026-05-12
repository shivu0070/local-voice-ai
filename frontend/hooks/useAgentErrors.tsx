import { useEffect } from 'react';
import { useAgent, useSessionContext } from '@livekit/components-react';
import { toastAlert } from '@/components/livekit/alert-toast';

export function useAgentErrors() {
  const agent = useAgent();
  const { isConnected, end } = useSessionContext();

  useEffect(() => {
    if (isConnected && agent.state === 'failed') {
      const reasons = agent.failureReasons;

      const isProd = process.env.NODE_ENV === 'production';

      if (isProd) {
        toastAlert({
          title: 'Session ended',
          description: (
            <>
              {reasons.length > 1 && (
                <ul className="list-inside list-disc">
                  {reasons.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
              )}
              {reasons.length === 1 && <p className="w-full">{reasons[0]}</p>}
              <p className="w-full">
                <a
                  target="_blank"
                  rel="noopener noreferrer"
                  href="https://docs.livekit.io/agents/start/voice-ai/"
                  className="whitespace-nowrap underline"
                >
                  See quickstart guide
                </a>
                .
              </p>
            </>
          ),
        });
        end();
      } else {
        // eslint-disable-next-line no-console
        console.warn('[LiveKit] agent failed; suppressing toast in dev', reasons);
      }
    }
  }, [agent, isConnected, end]);
}
