import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { DataWindow } from "./features/data/DataWindow";
import "./styles/tokens.css";
import "./styles/qt-base.css";
import "./styles/data.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <DataWindow />
  </StrictMode>,
);
