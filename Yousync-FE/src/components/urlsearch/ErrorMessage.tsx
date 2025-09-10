// src/components/urlsearch/ErrorMessage.tsx

interface ErrorMessageProps {
  message: string;
}

export default function ErrorMessage({ message }: ErrorMessageProps) {
  return (
    <div className="flex items-center justify-center py-20">
      <div className="text-red-400 font-medium">{message}</div>
    </div>
  );
}
