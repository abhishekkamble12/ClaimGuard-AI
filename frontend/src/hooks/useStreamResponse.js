import { useState, useRef, useCallback, useEffect } from 'react';
import { createResponseStream } from '../api/client';

/**
 * Hook for streaming AI-generated dispute response via SSE.
 * Returns { text, isStreaming, error, startStream, stopStream }
 */
export function useStreamResponse() {
  const [text, setText] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const sourceRef = useRef(null);

  const stopStream = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const startStream = useCallback((disputeId, apiKey = null) => {
    stopStream();
    setText('');
    setError(null);
    setIsStreaming(true);

    const source = createResponseStream(disputeId, apiKey);
    sourceRef.current = source;

    source.onmessage = (event) => {
      setText((prev) => prev + event.data);
    };

    source.onerror = (e) => {
      if (source.readyState === EventSource.CLOSED) {
        /* Stream completed normally */
        setIsStreaming(false);
      } else {
        setError('Stream connection lost. Please retry.');
        setIsStreaming(false);
      }
      source.close();
      sourceRef.current = null;
    };
  }, [stopStream]);

  /* Cleanup on unmount */
  useEffect(() => {
    return () => {
      if (sourceRef.current) {
        sourceRef.current.close();
      }
    };
  }, []);

  return { text, isStreaming, error, startStream, stopStream };
}
