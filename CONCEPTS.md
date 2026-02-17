# Explain like I'm 5: Advanced AI-Physics Concepts

This document explains the "Moonshot" ideas from the Research Roadmap without technical jargon.

## 1. Neural ODEs: "Infinite FPS"

*   **Current AI:** Predicts the future like a **flipbook** (Step 1 -> Step 2 -> Step 3). If you ask "Where is the planet at Step 1.5?", it has to guess (interpolate).
*   **Neural ODEs:** Predicts the future like a **continuous video**. It learns the *math equation of motion* itself (the "derivative").
*   **Benefit:** You can ask the AI for the state at *any* time (e.g., $t=3.14159$) and it gives a perfect answer instantly. It handles "chaos" much better because it understands time flow.

## 2. Symbolic Regression: "The AI Scientist"

*   **Current AI:** A "Black Box". It gives the right answer, but we don't know *how*. It's like a student who memorizes the textbook but can't explain the concepts.
*   **Symbolic Regression:** We force the AI to translate its internal weights into a **human-readable formula**.
*   **The Goal:** We want the AI to watch stars moving and output the text:
    $$ F = G \frac{m_1 m_2}{r^2} $$
    If it does this, it has **rediscovered Newton's Law of Gravity** from scratch. This is "AI-Driven Scientific Discovery".

## 3. Hydrodynamics (SPH): "Adding Water"

*   **Current AI:** Simulates **Stars** (billiard balls). Gravity pulls them together. It's a **"Dry"** simulation.
*   **SPH:** Adds **Gas** (water/clouds). Gas has pressure, viscosity (thickness), and temperature.
*   **Benefit:** When gas clouds crash together, they can heat up and collapse to form **New Stars**. This allows simulating the full lifecycle of a galaxy (like the Milky Way), moving from "physics demo" to "real astrophysics".
