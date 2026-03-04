---
name: remotion
description: "Create programmatic videos using React and Remotion.\n  MANDATORY TRIGGERS: video, Remotion, React video, animated video, MP4, WebM, programmatic video"
---

# Remotion Video Skill

## Quick Start

### Project Setup
```bash
npx create-video@latest my-video --template blank
cd my-video
npm install
```

### Basic Composition
```tsx
import { Composition } from 'remotion';
import { MyVideo } from './MyVideo';

export const RemotionRoot: React.FC = () => {
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

## Core Concepts

### Frames, Not Time
Everything is frame-based: `fps x seconds = frames`. At 30fps, a 5-second clip = 150 frames.

### Interpolation
```tsx
const opacity = interpolate(frame, [0, 30], [0, 1], { extrapolateRight: 'clamp' });
const x = interpolate(frame, [0, 60], [-100, 0]);
```

### Springs (Natural Motion)
```tsx
const scale = spring({ frame, fps, config: { damping: 200, mass: 0.5 } });
```

### Sequences (Timing Offsets)
```tsx
import { Sequence } from 'remotion';

<Sequence from={0} durationInFrames={60}><TitleCard /></Sequence>
<Sequence from={60} durationInFrames={90}><MainContent /></Sequence>
<Sequence from={150}><Outro /></Sequence>
```

### Data-Driven Content
```tsx
export const DataVideo: React.FC<{ items: string[] }> = ({ items }) => {
    const frame = useCurrentFrame();
    const currentIndex = Math.floor(frame / 30) % items.length;
    return <h1>{items[currentIndex]}</h1>;
};
```

## Advanced Patterns

### Audio
```tsx
import { Audio, staticFile } from 'remotion';
<Audio src={staticFile('music.mp3')} volume={0.5} />
```

### Images and Static Files
```tsx
import { Img, staticFile } from 'remotion';
<Img src={staticFile('logo.png')} style={{ width: 200 }} />
```

### Delay Render (Async Data)
```tsx
import { delayRender, continueRender } from 'remotion';

const [data, setData] = useState(null);
const [handle] = useState(() => delayRender());

useEffect(() => {
    fetch('/api/data').then(r => r.json()).then(d => {
        setData(d);
        continueRender(handle);
    });
}, []);
```

## Rendering

```bash
npx remotion preview                                          # Preview
npx remotion render src/index.ts MyVideo out/video.mp4        # MP4
npx remotion render src/index.ts MyVideo out/video.webm --codec vp8  # WebM
npx remotion render src/index.ts MyVideo out/video.mp4 --quality 80  # Custom quality
```

## Best Practices

- Keep compositions modular (one component per scene)
- Use `useCurrentFrame()` for all animation logic
- Pre-load assets with `staticFile()` or `delayRender()`
- Test in preview before rendering full video
- Use 30fps for most content, 60fps for smooth motion
- Keep video under 2 minutes for best performance

## Installation

```bash
npm init video -- --template blank
```
