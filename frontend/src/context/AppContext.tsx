"use client";

import {
  createContext,
  useContext,
  useReducer,
  type Dispatch,
  type ReactNode,
} from "react";
import type { AppState, ExplainResponse, Schema } from "@/types";

// ─── Actions ─────────────────────────────────────────────────────────────────

type Action =
  | { type: "UPLOAD_START" }
  | {
      type: "UPLOAD_SUCCESS";
      payload: { sessionId: string; schema: Schema; modelType: string; featureCount: number };
    }
  | { type: "UPLOAD_ERROR"; payload: string }
  | { type: "EXPLAIN_START" }
  | { type: "EXPLAIN_SUCCESS"; payload: ExplainResponse }
  | { type: "EXPLAIN_ERROR"; payload: string }
  | { type: "RESET" };

// ─── Reducer ─────────────────────────────────────────────────────────────────

const initialState: AppState = {
  sessionId: null,
  schema: null,
  modelType: null,
  featureCount: null,
  uploadStatus: "idle",
  explainResult: null,
  isExplaining: false,
  appErrors: [],
};

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "UPLOAD_START":
      return { ...state, uploadStatus: "uploading", appErrors: [] };
    case "UPLOAD_SUCCESS":
      return {
        ...state,
        uploadStatus: "success",
        sessionId: action.payload.sessionId,
        schema: action.payload.schema,
        modelType: action.payload.modelType,
        featureCount: action.payload.featureCount,
        explainResult: null,
        appErrors: [],
      };
    case "UPLOAD_ERROR":
      return { ...state, uploadStatus: "error", appErrors: [action.payload] };
    case "EXPLAIN_START":
      return { ...state, isExplaining: true, appErrors: [] };
    case "EXPLAIN_SUCCESS":
      return { ...state, isExplaining: false, explainResult: action.payload };
    case "EXPLAIN_ERROR":
      return { ...state, isExplaining: false, appErrors: [action.payload] };
    case "RESET":
      return initialState;
    default:
      return state;
  }
}

// ─── Context ─────────────────────────────────────────────────────────────────

interface AppContextValue {
  state: AppState;
  dispatch: Dispatch<Action>;
}

const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used within AppProvider");
  return ctx;
}
