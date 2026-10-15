import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { DialogProvider } from "./components/Dialogs";
import { StoreProvider } from "./state";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <DialogProvider>
      <StoreProvider>
        <App />
      </StoreProvider>
    </DialogProvider>
  </StrictMode>,
);
