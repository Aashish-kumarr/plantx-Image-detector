/**
 * PlantDx Centralized Frontend API Client.
 *
 * Connects Next.js frontend to FastAPI backend endpoints.
 * Handles:
 *  - Base URL resolution from NEXT_PUBLIC_API_BASE_URL (defaults to http://localhost:8000)
 *  - Authorization Bearer header injection via token provider
 *  - Standardized backend error envelope parsing ({ error: { code, message } })
 *  - Network error handling
 *  - Multipart image uploads
 */

const API_BASE_URL = (
  "http://localhost:8000" || "http://localhost:8000"
).replace(/\/$/, "");

let tokenProvider = null;

/**
 * Register a function that returns a promise resolving to the Firebase ID token.
 * @param {Function} provider () => Promise<string | null>
 */
export function setTokenProvider(provider) {
  tokenProvider = provider;
}

export class ApiError extends Error {
  constructor(code, message, status) {
    super(message);
    this.name = "ApiError";
    this.code = code || "API_ERROR";
    this.status = status || 500;
  }
}

/**
 * Execute an HTTP request against the FastAPI v1 backend.
 *
 * @param {string} endpoint - Path relative to /api/v1 (e.g. '/images')
 * @param {Object} options - Fetch options (method, body, headers, isMultipart)
 * @returns {Promise<Object>} JSON response payload
 */
export async function apiRequest(endpoint, options = {}) {
  const {
    method = "GET",
    body = null,
    headers = {},
    isMultipart = false,
  } = options;

  const requestHeaders = { ...headers };

  // Inject Authorization Bearer token if tokenProvider registered
  if (tokenProvider && typeof tokenProvider === "function") {
    try {
      const token = await tokenProvider();
      if (token) {
        requestHeaders["Authorization"] = `Bearer ${token}`;
      }
    } catch (err) {
      console.warn("Failed to retrieve auth token for request:", err);
    }
  }

  if (!isMultipart && body && typeof body === "object") {
    requestHeaders["Content-Type"] = "application/json";
  }

  const url = `${API_BASE_URL}/api/v1${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;

  let response;
  try {
    response = await fetch(url, {
      method,
      headers: requestHeaders,
      body: isMultipart ? body : (body && typeof body === "object" ? JSON.stringify(body) : body),
    });
  } catch (err) {
    throw new ApiError(
      "NETWORK_ERROR",
      "Unable to connect to PlantDx backend service. Please check your network connection.",
      0
    );
  }

  let data;
  try {
    data = await response.json();
  } catch (err) {
    if (!response.ok) {
      throw new ApiError(
        "SERVER_ERROR",
        `Backend request failed with status ${response.status}`,
        response.status
      );
    }
    return {};
  }

  if (!response.ok) {
    if (data && data.error) {
      throw new ApiError(
        data.error.code || "REQUEST_FAILED",
        data.error.message || "An error occurred during the request.",
        response.status
      );
    }
    throw new ApiError(
      "REQUEST_FAILED",
      `Request failed with status ${response.status}`,
      response.status
    );
  }

  return data;
}
