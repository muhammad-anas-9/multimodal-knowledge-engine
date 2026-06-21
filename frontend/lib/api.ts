const API_BASE = "http://localhost:8000";

export interface ChatResponse {
  text: string;
  thought_process?: string;
  tokens_used_this_turn: number;
}

export interface UploadResponse {
  status: string;
  filename: string;
}

export interface ClearResponse {
  status: string;
}

export async function sendChatMessage(
  message: string,
  modelId: string,
  useRag: boolean,
  deviceTokens: number,
  deviceId: string
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      model_id: modelId,
      use_rag: useRag,
      current_device_tokens: deviceTokens,
      device_id: deviceId,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export async function uploadDocument(file: File, deviceId: string): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("device_id", deviceId);

  const response = await fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export async function clearKnowledgeBase(deviceId: string): Promise<ClearResponse> {
  const response = await fetch(`${API_BASE}/api/clear`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      device_id: deviceId,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}
