import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { AgentsWindow } from "./features/agents/AgentsWindow";
import "./styles/tokens.css";
import "./styles/slice.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AgentsWindow />
  </StrictMode>,
);
