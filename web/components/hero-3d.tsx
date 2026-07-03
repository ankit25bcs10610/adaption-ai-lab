"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import { useReducedMotion } from "framer-motion";
import * as THREE from "three";
import { useAccent } from "@/components/use-accent";
import { accentById } from "@/lib/themes";

/** The model core: a wireframe icosahedron (primary accent) + inner shell (secondary accent). */
function Core({ animate, colors }: { animate: boolean; colors: readonly string[] }) {
  const wire = useRef<THREE.LineSegments>(null);
  const inner = useRef<THREE.Mesh>(null);

  const wireGeo = useMemo(() => {
    const g = new THREE.IcosahedronGeometry(4.2, 1);
    return new THREE.WireframeGeometry(g);
  }, []);
  const innerGeo = useMemo(() => new THREE.IcosahedronGeometry(2.4, 0), []);

  useFrame((_, dt) => {
    if (!animate) return;
    if (wire.current) {
      wire.current.rotation.y += dt * 0.1;
      wire.current.rotation.x += dt * 0.05;
    }
    if (inner.current) {
      inner.current.rotation.y -= dt * 0.16;
      inner.current.rotation.z += dt * 0.09;
    }
  });

  return (
    <group>
      <lineSegments ref={wire} geometry={wireGeo}>
        <lineBasicMaterial color={colors[0]} transparent opacity={0.55} />
      </lineSegments>
      <mesh ref={inner} geometry={innerGeo}>
        <meshBasicMaterial color={colors[1]} wireframe transparent opacity={0.25} />
      </mesh>
    </group>
  );
}

/** Orbiting particle field = the tools. */
function Tools({ animate, color }: { animate: boolean; color: string }) {
  const ref = useRef<THREE.Points>(null);
  const positions = useMemo(() => {
    const N = 260;
    const arr = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const r = 6 + Math.random() * 6;
      const t = Math.random() * Math.PI * 2;
      const p = Math.acos(2 * Math.random() - 1);
      arr[i * 3] = r * Math.sin(p) * Math.cos(t);
      arr[i * 3 + 1] = r * Math.sin(p) * Math.sin(t);
      arr[i * 3 + 2] = r * Math.cos(p);
    }
    return arr;
  }, []);

  useFrame((_, dt) => {
    if (animate && ref.current) ref.current.rotation.y += dt * 0.06;
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial color={color} size={0.07} transparent opacity={0.8} sizeAttenuation />
    </points>
  );
}

/** Camera parallax toward the pointer. */
function Parallax({ animate }: { animate: boolean }) {
  const { camera, pointer } = useThree();
  useFrame(() => {
    if (!animate) return;
    camera.position.x += (pointer.x * 4 - camera.position.x) * 0.04;
    camera.position.y += (pointer.y * 4 - camera.position.y) * 0.04;
    camera.lookAt(0, 0, 0);
  });
  return null;
}

export function Hero3D() {
  const reduce = useReducedMotion();
  const { accent } = useAccent();
  const colors = accentById(accent).colors;
  // Honor a reduced-motion preference by not mounting a WebGL context at all — the global aurora +
  // grid-fade still provide ambiance, and we save a GPU context on low-power / motion-sensitive setups.
  if (reduce) return null;
  return (
    <Canvas
      className="absolute inset-0"
      dpr={[1, 2]}
      camera={{ position: [0, 0, 15], fov: 60 }}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      aria-hidden
    >
      <Core animate colors={colors} />
      <Tools animate color={colors[2]} />
      <Parallax animate />
    </Canvas>
  );
}
