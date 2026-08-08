import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { AgentsWindow } from "./features/agents/AgentsWindow";
import "./styles/tokens.css";
import "./styles/agents.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AgentsWindow />
  </StrictMode>,
);
