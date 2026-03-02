---
name: remotion
description: "Create programmatic videos using React and Remotion. For video content, animated videos, data-driven video, and video templates."
---

# Remotion Video Skill

MANDATORY TRIGGERS: video, animation, render, mp4, webm, remotion, programmatic video

## Technology Stack

- **Framework**: Remotion (React-based video creation)
- **Runtime**: Node.js 18+
- **Rendering**: Chromium headless via `@remotion/renderer`

## Quick Start

### Project Setup
```bash
npx create-video@latest my-video --template blank
cd my-video
npm install
```

### Basic Composition
```tsx
// src/Composition.tsx
import { Composition } from 'remotion';
import { MyVideo } from './MyVideo';

export const RemotionRoot = () => {
    return (
        <Composition
            id="MyVideo"
            component={MyVideo}
            durationInFrames={150}
            fps={30}
            width={1920}
            height={1080}
        />
    );
};
```

### Basic Video Component
```tsx
// src/MyVideo.tsx
import { useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';

export const MyVideo = () => {
    const frame = useCurrentFrame();
    const { fps } = useVideoConfig();

    const opacity = interpolate(frame, [0, 30], [0, 1], {
        extrapolateRight: 'clamp',
    });

    const scale = spring({ frame, fps, config: { damping: 200 } });

    return (
        <div style={{ flex: 1, justifyContent: 'center', alignItems: 'center', opacity }}>
            <h1 style={{ transform: `scale(${scale})` }}>Hello World</h1>
        </div>
    );
};
```

## Rendering

```bash
# Preview
npx remotion preview

# Render MP4
npx remotion render src/index.ts MyVideo out/video.mp4

# Render with custom settings
npx remotion render src/index.ts MyVideo out/video.mp4 --codec h264 --quality 80
```

## Key Concepts

1. **Frames, not time**: Everything is frame-based (fps × seconds = frames)
2. **Interpolation**: Use `interpolate()` for smooth transitions
3. **Springs**: Use `spring()` for natural motion
4. **Sequences**: Use `<Sequence>` to offset timing of elements
5. **Data-driven**: Pass props to compositions for dynamic content

## Best Practices

- Keep compositions modular (one component per scene)
- Use `useCurrentFrame()` for all animation logic
- Pre-load assets with `staticFile()` or `delayRender()`
- Test in preview before rendering full video
- Use 30fps for most content, 60fps for smooth motion

## Installation

```bash
npm init video -- --template blank
```
