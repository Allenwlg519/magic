/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef, useState } from 'react';
import p5 from 'p5';
import { ObjectDetector, HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

const FLOWER_COLORS = [
  [330, 40, 100], // 梦幻粉
  [280, 30, 100], // 梦幻紫
  [200, 30, 100], // 梦幻蓝
  [180, 40, 100], // 梦幻青
  [300, 20, 100], // 珍珠白
];

const METEOR_COLORS = [
  [45, 90, 100],   // 金黄色
  [55, 100, 100],  // 明黄色
  [40, 80, 100],   // 琥珀色
];

export default function App() {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    let objectDetector: ObjectDetector | null = null;
    let handLandmarker: HandLandmarker | null = null;
    let lastVideoTime = -1;
    let results: any = null;
    let handResults: any = null;
    let handsMerged = false;
    let lastMergeTime = 0;

    const initMediaPipe = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm"
        );
        
        // 初始化物体识别
        objectDetector = await ObjectDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          scoreThreshold: 0.2,
        });

        // 初始化手部识别
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2
        });

        setIsLoading(false);
      } catch (err) {
        console.error("MediaPipe initialization failed:", err);
        setError(`无法加载识别模型 (MediaPipe): ${err instanceof Error ? err.message : '网络连接失败'}。请检查网络或尝试刷新页面。`);
      }
    };

    initMediaPipe();

    const sketch = (p: p5) => {
      let flowers: GrowingFlower[] = [];
      let groundFlowers: GroundFlower[] = [];
      let meteors: Meteor[] = [];
      let butterflies: Butterfly[] = [];
      let particles: MagicParticle[] = [];
      let canvas: p5.Renderer;
      let detectedObjects: any[] = [];
      let detectedHands: any[] = [];
      let prevHandPos: p5.Vector[] = [];
      let swayFactor = 0;

      p.setup = () => {
        canvas = p.createCanvas(p.windowWidth, p.windowHeight);
        p.colorMode(p.HSB, 360, 100, 100, 1);
        
        // 设置视频流
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
          navigator.mediaDevices.getUserMedia({ 
            video: { 
              width: { ideal: 1280 },
              height: { ideal: 720 }
            } 
          }).then((stream) => {
            if (videoRef.current) {
              videoRef.current.srcObject = stream;
              // 确保视频元数据加载后再播放
              videoRef.current.onloadedmetadata = () => {
                videoRef.current?.play();
              };
            }
          }).catch(err => {
            console.error("Camera access denied:", err);
            if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
              setError("摄像头访问被拒绝。请在浏览器地址栏点击摄像头图标并选择“允许”，然后刷新页面。");
            } else {
              setError(`无法访问摄像头: ${err.message || '未知错误'}`);
            }
          });
        }
      };

      p.draw = () => {
        p.clear();
        
        const video = videoRef.current;
        const isVideoReady = !!(
          video && 
          video instanceof HTMLVideoElement && 
          video.readyState >= 2 && 
          video.videoWidth > 0 && 
          video.videoHeight > 0
        );

        // 1. 处理物体识别
        if (isVideoReady) {
          try {
            let startTimeMs = performance.now();
            if (lastVideoTime !== video.currentTime) {
              lastVideoTime = video.currentTime;
              
              if (objectDetector) {
                results = objectDetector.detectForVideo(video, startTimeMs);
                detectedObjects = results.detections || [];
              }
              
              if (handLandmarker) {
                handResults = handLandmarker.detectForVideo(video, startTimeMs);
                detectedHands = handResults.landmarks || [];
              }
            }
          } catch (err) {
            console.error("Detection error:", err);
          }
        }

        // 2. 绘制视频背景 (镜像)
        let dw = p.width;
        let dh = p.height;
        let offsetX = 0;
        let offsetY = 0;

        if (isVideoReady) {
          const ctx = (p as any).drawingContext as CanvasRenderingContext2D;
          if (ctx) {
            ctx.save();
            ctx.translate(p.width, 0);
            ctx.scale(-1, 1);
            
            let vRatio = video.videoWidth / video.videoHeight;
            let cRatio = p.width / p.height;
            
            if (vRatio > cRatio) {
              dh = p.height;
              dw = p.height * vRatio;
            } else {
              dw = p.width;
              dh = p.width / vRatio;
            }
            
            offsetX = (p.width - dw) / 2;
            offsetY = (p.height - dh) / 2;
            
            ctx.globalAlpha = 1.0;
            ctx.drawImage(video, offsetX, offsetY, dw, dh);
            ctx.restore();
          }
        }

        // 3. 处理开花逻辑 & 调试显示
        if (detectedObjects.length > 0) {
          const vWidth = video?.videoWidth || 1;
          const vHeight = video?.videoHeight || 1;

          // 标记当前帧识别到的杯子位置
          const currentCups: {x: number, y: number, w: number, id: string}[] = [];

          for (const detection of detectedObjects) {
            const category = detection.categories[0].categoryName.toLowerCase();
            const score = detection.categories[0].score;
            const box = detection.boundingBox;
            
            // 识别杯子及类似物体
            const targetCategories = ['cup', 'wine glass', 'mug', 'bottle', 'bowl', 'vase', 'can'];
            if (box && targetCategories.includes(category)) {
              const vx = box.originX + box.width / 2;
              const vy = box.originY;
              
              const mappedX = offsetX + vx * (dw / vWidth);
              const mappedY = offsetY + vy * (dh / vHeight);
              
              const centerX = p.width - mappedX;
              const topY = mappedY;
              
              currentCups.push({ 
                x: centerX, 
                y: topY, 
                w: box.width * (dw / vWidth),
                id: `${category}_${Math.round(box.originX)}` 
              });

              // 调试：绘制识别框
              const canvasRectX = p.width - (offsetX + (box.originX + box.width) * (dw / vWidth));
              const canvasRectY = offsetY + box.originY * (dh / vHeight);
              p.noFill();
              p.stroke(120, 80, 100, 0.3); // 降低调试框亮度
              p.strokeWeight(1);
              p.rect(canvasRectX, canvasRectY, box.width * (dw / vWidth), box.height * (dh / vHeight));
            }
          }

          // 更新或创建花朵 (茂密模式：每个杯子对应多朵花)
          for (const cup of currentCups) {
            // 检查该杯子附近已有的花朵数量
            let nearbyFlowers = flowers.filter(f => !f.isFalling && p.dist(cup.x, cup.y, f.pos.x, f.pos.y) < cup.w / 2);
            
            // 如果花朵不够茂密，则在杯口范围内随机生成
            if (nearbyFlowers.length < 12 && flowers.length < 200) {
              const offsetX = p.random(-cup.w / 3, cup.w / 3);
              flowers.push(new GrowingFlower(p, cup.x + offsetX, cup.y));
            }

            // 更新已有花朵的目标位置，让它们跟随杯子
            for (let flower of nearbyFlowers) {
              flower.updateTarget(cup.x + (flower.pos.x - cup.x), cup.y);
            }
          }
        }

        // 4. 处理手部魔法 (流星 & 颗粒 & 捏合)
        if (detectedHands.length >= 2) {
          const h1 = detectedHands[0][0];
          const h2 = detectedHands[1][0];
          const d = p.dist(h1.x, h1.y, h2.x, h2.y);
          
          if (d < 0.15) {
            handsMerged = true;
            lastMergeTime = p.millis();
            // 产生汇聚光点 (金色仙粉)
            const midX = p.width - (offsetX + (h1.x + h2.x) / 2 * dw);
            const midY = offsetY + (h1.y + h2.y) / 2 * dh;
            for(let i=0; i<3; i++) particles.push(new MagicParticle(p, midX, midY, true));
          } else if (handsMerged && d > 0.3) {
            // 手心合拢后打开，释放蝴蝶
            const midX = p.width - (offsetX + (h1.x + h2.x) / 2 * dw);
            const midY = offsetY + (h1.y + h2.y) / 2 * dh;
            butterflies.push(new Butterfly(p, midX, midY));
            handsMerged = false;
            // 爆发光效 (金色仙粉)
            for(let i=0; i<20; i++) particles.push(new MagicParticle(p, midX, midY, true));
          }
        }

        if (detectedHands.length > 0) {
          let totalHandX = 0;
          for (let i = 0; i < detectedHands.length; i++) {
            const hand = detectedHands[i];
            const wrist = hand[0];
            const mappedX = p.width - (offsetX + wrist.x * dw);
            const mappedY = offsetY + wrist.y * dh;
            const currentPos = p.createVector(mappedX, mappedY);

            // 产生随动作掉落的魔法颗粒 (金色仙粉)
            if (prevHandPos[i]) {
              const dist = p.dist(currentPos.x, currentPos.y, prevHandPos[i].x, prevHandPos[i].y);
              if (dist > 5) {
                for (let j = 0; j < 2; j++) {
                  particles.push(new MagicParticle(p, mappedX, mappedY, true));
                }
              }
            }
            prevHandPos[i] = currentPos;
            totalHandX += mappedX;

            // 判断是否张开手掌 (流星) - 增加 handsMerged 检查
            const isPalmOpen = (p as any).checkPalmOpen(hand);
            if (isPalmOpen && !handsMerged) {
              const palmX = (hand[0].x + hand[5].x + hand[17].x) / 3;
              const palmY = (hand[0].y + hand[5].y + hand[17].y) / 3;
              const centerX = p.width - (offsetX + palmX * dw);
              const topY = offsetY + palmY * dh;
              for (let j = 0; j < 2; j++) {
                meteors.push(new Meteor(p, centerX, topY));
              }
            }

            // 判断是否捏合 (捏合产生地上的花)
            const isPinching = (p as any).checkPinch(hand);
            if (isPinching) {
              const indexTip = hand[8];
              const thumbTip = hand[4];
              const pinchX = p.width - (offsetX + (indexTip.x + thumbTip.x) / 2 * dw);
              // 在屏幕下方对应位置长出花
              if (p.frameCount % 10 === 0) {
                groundFlowers.push(new GroundFlower(p, pinchX, p.height - 10));
              }
            }
          }
          // 计算摇晃因子 (基于手部平均水平位置的变化)
          const avgHandX = totalHandX / detectedHands.length;
          swayFactor = p.map(avgHandX, 0, p.width, -1, 1);
        } else {
          swayFactor = p.lerp(swayFactor, 0, 0.05);
        }

        // 更新和绘制蝴蝶
        for (let i = butterflies.length - 1; i >= 0; i--) {
          butterflies[i].update(particles);
          butterflies[i].display();
          if (butterflies[i].isDead()) butterflies.splice(i, 1);
        }

        // 更新和绘制魔法颗粒
        for (let i = particles.length - 1; i >= 0; i--) {
          particles[i].update();
          particles[i].display();
          if (particles[i].isDead()) particles.splice(i, 1);
        }

        // 更新和绘制地面花朵
        for (let i = groundFlowers.length - 1; i >= 0; i--) {
          groundFlowers[i].update(swayFactor);
          groundFlowers[i].display();
          if (groundFlowers[i].isDead()) groundFlowers.splice(i, 1);
        }

        // 更新和绘制花朵 (杯子上的)
        for (let i = flowers.length - 1; i >= 0; i--) {
          flowers[i].update();
          flowers[i].display();
          if (flowers[i].isDead()) {
            flowers.splice(i, 1);
          }
        }

        // 更新和绘制流星
        for (let i = meteors.length - 1; i >= 0; i--) {
          meteors[i].update(particles);
          meteors[i].display();
          if (meteors[i].isDead()) {
            meteors.splice(i, 1);
          }
        }
      };

      // 检查手掌是否完全张开
      (p as any).checkPalmOpen = (hand: any[]) => {
        const wrist = hand[0];
        const fingers = [4, 8, 12, 16, 20]; // 指尖 (含大拇指)
        const bases = [2, 5, 9, 13, 17];    // 指根
        let openCount = 0;
        for (let i = 0; i < fingers.length; i++) {
          const tipDist = p.dist(wrist.x, wrist.y, hand[fingers[i]].x, hand[fingers[i]].y);
          const baseDist = p.dist(wrist.x, wrist.y, hand[bases[i]].x, hand[bases[i]].y);
          // 增加判断阈值，确保手指完全伸直
          if (tipDist > baseDist * 1.2) openCount++;
        }
        return openCount === 5; // 必须 5 根手指全部张开
      };

      // 检查是否捏合 (大拇指和食指)
      (p as any).checkPinch = (hand: any[]) => {
        const thumbTip = hand[4];
        const indexTip = hand[8];
        const distance = p.dist(thumbTip.x, thumbTip.y, indexTip.x, indexTip.y);
        return distance < 0.05;
      };

      p.windowResized = () => {
        p.resizeCanvas(p.windowWidth, p.windowHeight);
      };
    };

    class GrowingFlower {
      p: p5;
      pos: p5.Vector;
      targetPos: p5.Vector;
      vel: p5.Vector;
      growth: number;
      maxGrowth: number;
      hue: number;
      petals: number;
      life: number;
      maxLife: number = 60; 
      isWithered: boolean = false;
      isFalling: boolean = false;
      stemPoints: p5.Vector[] = [];
      fallTimer: number;
      shapeType: number;

      constructor(p: p5, x: number, y: number) {
        this.p = p;
        this.pos = p.createVector(x, y);
        this.targetPos = p.createVector(x, y);
        this.vel = p.createVector(0, 0);
        this.growth = 0;
        this.maxGrowth = p.random(80, 140); // 进一步增大花朵
        this.life = this.maxLife;
        this.fallTimer = p.random(150, 300);
        this.shapeType = p.floor(p.random(3)); // 随机形状类型
        
        const color = p.random(FLOWER_COLORS);
        this.hue = color[0];
        this.petals = p.floor(p.random(6, 12)); // 更多花瓣

        for (let i = 0; i < 5; i++) {
          this.stemPoints.push(p.createVector(p.random(-5, 5), -i * 10));
        }
      }

      updateTarget(x: number, y: number) {
        if (!this.isFalling) {
          this.targetPos.set(x, y);
          this.life = this.maxLife;
          this.isWithered = false;
        }
      }

      update() {
        if (this.isFalling) {
          this.vel.y += 0.1; // 重力
          this.pos.add(this.vel);
          this.growth -= 0.005; // 掉落时慢慢枯萎
        } else {
          this.pos.x = this.p.lerp(this.pos.x, this.targetPos.x, 0.1);
          this.pos.y = this.p.lerp(this.pos.y, this.targetPos.y, 0.1);

          if (this.life > 0) {
            this.life--;
            if (this.growth < 1) this.growth += 0.02;
            
            this.fallTimer--;
            if (this.fallTimer <= 0) {
              this.isFalling = true;
              this.vel = this.p.createVector(this.p.random(-1, 1), this.p.random(-2, 0));
            }
          } else {
            this.isWithered = true;
            if (this.growth > 0) this.growth -= 0.02;
          }
        }
      }

      display() {
        if (this.growth <= 0) return;

        this.p.push();
        this.p.translate(this.pos.x, this.pos.y);
        
        const currentHeight = this.growth * this.maxGrowth;
        
        // 掉落时不画茎
        if (!this.isFalling) {
          this.p.noFill();
          this.p.stroke(120, 60, 60, this.growth);
          this.p.strokeWeight(2 * this.growth);
          this.p.beginShape();
          this.p.vertex(0, 0);
          for (let i = 1; i < this.stemPoints.length; i++) {
            const pt = this.stemPoints[i];
            this.p.curveVertex(pt.x * this.growth, pt.y * this.growth * (this.maxGrowth/40));
          }
          this.p.endShape();
        }

        const flowerY = this.isFalling ? 0 : -currentHeight;
        this.p.translate(0, flowerY);
        this.p.rotate(this.p.frameCount * 0.02 + (this.isFalling ? this.p.frameCount * 0.05 : 0));
        
        const flowerSize = 20 * this.growth;
        this.drawFlowerHead(flowerSize);
        
        this.p.pop();
      }

      drawFlowerHead(size: number) {
        this.p.noStroke();
        // 绘制花瓣，使用渐变感
        for (let i = 0; i < this.petals; i++) {
          this.p.push();
          this.p.rotate((this.p.TWO_PI / this.petals) * i);
          
          // 渐变效果：多层绘制
          for (let j = 0; j < 3; j++) {
            const layerSize = size * (1 - j * 0.2);
            const layerBright = 100 - j * 5;
            this.p.fill(this.hue, 30 - j * 5, layerBright, this.growth * 0.5); // 梦幻透明
            
            this.p.beginShape();
            if (this.shapeType === 0) {
              // 尖瓣
              this.p.vertex(0, 0);
              this.p.bezierVertex(layerSize * 0.5, -layerSize * 0.3, layerSize * 0.8, 0, layerSize * 0.5, layerSize * 0.3);
            } else if (this.shapeType === 1) {
              // 圆瓣
              this.p.ellipse(layerSize / 2, 0, layerSize, layerSize * 0.6);
            } else {
              // 心形/复瓣
              this.p.vertex(0, 0);
              this.p.bezierVertex(layerSize * 0.4, -layerSize * 0.6, layerSize * 1.2, -layerSize * 0.2, layerSize * 0.5, 0);
              this.p.bezierVertex(layerSize * 1.2, layerSize * 0.2, layerSize * 0.4, layerSize * 0.6, 0, 0);
            }
            this.p.endShape(this.p.CLOSE);
          }
          this.p.pop();
        }
        // 花蕊
        this.p.fill(60, 90, 100, this.growth * 0.8);
        this.p.ellipse(0, 0, size * 0.4);
        this.p.fill(45, 100, 100, this.growth * 0.8);
        this.p.ellipse(0, 0, size * 0.2);
      }

      isDead() {
        return (this.isWithered || this.isFalling) && this.growth <= 0;
      }
    }

    class GroundFlower {
      p: p5;
      pos: p5.Vector;
      growth: number;
      maxGrowth: number;
      hue: number;
      petals: number;
      sway: number = 0;
      life: number = 1.0;
      shapeType: number;

      constructor(p: p5, x: number, y: number) {
        this.p = p;
        this.pos = p.createVector(x, y);
        this.growth = 0;
        this.maxGrowth = p.random(100, 180); // 更大的地面花
        this.shapeType = p.floor(p.random(3));
        const color = p.random(FLOWER_COLORS);
        this.hue = color[0];
        this.petals = p.floor(p.random(6, 12));
      }

      update(swayFactor: number) {
        if (this.growth < 1) this.growth += 0.05;
        this.sway = this.p.lerp(this.sway, swayFactor * 20, 0.1);
        this.life -= 0.002; // 地面花朵寿命较长
      }

      display() {
        this.p.push();
        this.p.translate(this.pos.x, this.pos.y);
        
        const h = this.growth * this.maxGrowth;
        
        // 绘制茎
        this.p.stroke(120, 60, 60, this.life);
        this.p.strokeWeight(3);
        this.p.noFill();
        this.p.bezier(0, 0, this.sway/2, -h/3, this.sway, -h/2, this.sway, -h);
        
        // 绘制花头
        this.p.translate(this.sway, -h);
        this.p.rotate(this.p.frameCount * 0.02);
        
        const size = 35 * this.growth; // 增大尺寸
        this.p.noStroke();
        for (let i = 0; i < this.petals; i++) {
          this.p.push();
          this.p.rotate((this.p.TWO_PI / this.petals) * i);
          // 灵动花瓣
          this.p.fill(this.hue, 40, 100, this.life * 0.5); // 梦幻透明
          
          this.p.beginShape();
          if (this.shapeType === 0) {
            this.p.vertex(0, 0);
            this.p.bezierVertex(size * 0.6, -size * 0.4, size, 0, size * 0.6, size * 0.4);
          } else if (this.shapeType === 1) {
            this.p.ellipse(size / 2, 0, size, size * 0.7);
          } else {
            this.p.vertex(0, 0);
            this.p.bezierVertex(size * 0.5, -size * 0.8, size * 1.5, 0, size * 0.5, size * 0.8);
          }
          this.p.endShape(this.p.CLOSE);
          this.p.pop();
        }
        this.p.fill(60, 90, 100, this.life);
        this.p.ellipse(0, 0, size * 0.4);
        
        this.p.pop();
      }

      isDead() {
        return this.life <= 0;
      }
    }

    class MagicParticle {
      p: p5;
      pos: p5.Vector;
      vel: p5.Vector;
      size: number;
      hue: number;
      life: number;
      isFairyDust: boolean;

      constructor(p: p5, x: number, y: number, isFairyDust: boolean = false) {
        this.p = p;
        this.pos = p.createVector(x, y);
        this.isFairyDust = isFairyDust;
        
        if (isFairyDust) {
          this.vel = p.createVector(p.random(-0.5, 0.5), p.random(0.5, 2));
          this.size = p.random(2, 5);
          this.hue = p.random(40, 60); // 金色/黄色
        } else {
          this.vel = p.createVector(p.random(-1, 1), p.random(1, 3));
          this.size = p.random(4, 8);
          this.hue = p.random(360);
        }
        this.life = 1.0;
      }

      update() {
        this.pos.add(this.vel);
        this.life -= this.isFairyDust ? 0.015 : 0.02;
      }

      display() {
        this.p.noStroke();
        if (this.isFairyDust) {
          // 闪烁效果
          const sparkle = this.p.sin(this.p.frameCount * 0.5) * 0.5 + 0.5;
          this.p.fill(this.hue, 80, 100, this.life * sparkle);
        } else {
          this.p.fill(this.hue, 80, 100, this.life);
        }
        this.p.ellipse(this.pos.x, this.pos.y, this.size);
        
        const ctx = (this.p as any).drawingContext as CanvasRenderingContext2D;
        if (ctx) {
          ctx.shadowBlur = this.isFairyDust ? 15 : 10;
          ctx.shadowColor = this.p.color(this.hue, 100, 100, this.life).toString();
        }
      }

      isDead() {
        return this.life <= 0;
      }
    }

    class Meteor {
      p: p5;
      pos: p5.Vector;
      vel: p5.Vector;
      hue: number;
      size: number;
      life: number;
      history: p5.Vector[] = [];

      constructor(p: p5, x: number, y: number) {
        this.p = p;
        this.pos = p.createVector(x, y);
        this.vel = p.createVector(p.random(-8, 8), p.random(-8, 8));
        const color = p.random(METEOR_COLORS);
        this.hue = color[0];
        this.size = p.random(12, 24); 
        this.life = 1.0;
      }

      update(particles: MagicParticle[]) {
        this.history.push(this.pos.copy());
        if (this.history.length > 10) this.history.shift();

        this.pos.add(this.vel);
        this.vel.y += 0.15; 
        this.life -= 0.03; // 消失速度加快

        // 洒落金色仙粉
        if (this.p.frameCount % 2 === 0) {
          particles.push(new MagicParticle(this.p, this.pos.x, this.pos.y, true));
        }
      }

      display() {
        this.p.push();
        
        // 绘制拖尾
        this.p.noFill();
        for (let i = 0; i < this.history.length; i++) {
          const alpha = this.p.map(i, 0, this.history.length, 0, this.life * 0.5);
          const s = this.p.map(i, 0, this.history.length, 0, this.size);
          this.p.fill(this.hue, 60, 100, alpha);
          this.p.ellipse(this.history[i].x, this.history[i].y, s);
        }

        this.p.translate(this.pos.x, this.pos.y);
        this.p.noStroke();
        
        // 绘制星形
        this.p.fill(this.hue, 80, 100, this.life);
        this.drawStar(0, 0, this.size, this.size / 2, 5);
        
        // 发光效果
        const ctx = (this.p as any).drawingContext as CanvasRenderingContext2D;
        if (ctx) {
          ctx.shadowBlur = 25;
          ctx.shadowColor = this.p.color(this.hue, 100, 100, this.life).toString();
        }
        
        this.p.pop();
      }

      drawStar(x: number, y: number, radius1: number, radius2: number, npoints: number) {
        let angle = this.p.TWO_PI / npoints;
        let halfAngle = angle / 2.0;
        this.p.beginShape();
        for (let a = 0; a < this.p.TWO_PI; a += angle) {
          let sx = x + this.p.cos(a) * radius2;
          let sy = y + this.p.sin(a) * radius2;
          this.p.vertex(sx, sy);
          sx = x + this.p.cos(a + halfAngle) * radius1;
          sy = y + this.p.sin(a + halfAngle) * radius1;
          this.p.vertex(sx, sy);
        }
        this.p.endShape(this.p.CLOSE);
      }

      isDead() {
        return this.life <= 0;
      }
    }

    class Butterfly {
      p: p5;
      pos: p5.Vector;
      vel: p5.Vector;
      hue: number;
      life: number = 1.0;
      wingAngle: number = 0;
      size: number;

      constructor(p: p5, x: number, y: number) {
        this.p = p;
        this.pos = p.createVector(x, y);
        this.vel = p.createVector(p.random(-3, 3), p.random(-4, -2));
        this.hue = p.random(360); // 颜色完全随机
        this.size = p.random(40, 60); // 进一步增大蝴蝶
      }

      update(particles: MagicParticle[]) {
        this.pos.add(this.vel);
        this.vel.x += this.p.sin(this.p.frameCount * 0.1) * 0.15;
        this.wingAngle = this.p.sin(this.p.frameCount * 0.3) * 1.0;
        this.life -= 0.004;

        // 洒落粉末
        if (this.p.frameCount % 8 === 0) {
          particles.push(new MagicParticle(this.p, this.pos.x, this.pos.y, true));
        }
      }

      display() {
        this.p.push();
        this.p.translate(this.pos.x, this.pos.y);
        this.p.rotate(this.vel.heading() + this.p.HALF_PI);
        
        const ctx = (this.p as any).drawingContext as CanvasRenderingContext2D;
        if (ctx) {
          ctx.shadowBlur = 30;
          ctx.shadowColor = this.p.color(this.hue, 100, 100, this.life).toString();
        }

        this.p.noStroke();
        
        // 绘制更像 🦋 的翅膀
        for (let side of [-1, 1]) {
          this.p.push();
          this.p.scale(side, 1);
          this.p.rotate(this.wingAngle * side);
          
          // 上大翅膀 (更圆润且向上扬)
          this.p.fill(this.hue, 70, 100, this.life * 0.9);
          this.p.beginShape();
          this.p.vertex(0, 0);
          this.p.bezierVertex(-this.size * 0.5, -this.size * 1.2, -this.size * 2.2, -this.size * 0.8, -this.size * 1.8, 0);
          this.p.bezierVertex(-this.size * 1.5, this.size * 0.4, -this.size * 0.5, this.size * 0.2, 0, 0);
          this.p.endShape(this.p.CLOSE);
          
          // 下小翅膀 (更圆且向后方)
          this.p.fill(this.hue, 60, 100, this.life * 0.7);
          this.p.beginShape();
          this.p.vertex(0, 0);
          this.p.bezierVertex(-this.size * 0.2, this.size * 0.5, -this.size * 1.5, this.size * 1.8, -this.size * 1.2, this.size * 0.5);
          this.p.bezierVertex(-this.size * 0.8, this.size * 0.2, -this.size * 0.3, 0.1, 0, 0);
          this.p.endShape(this.p.CLOSE);

          // 翅膀花纹 (线条感)
          this.p.stroke(255, this.life * 0.3);
          this.p.strokeWeight(1);
          this.p.line(0, 0, -this.size * 1.5, -this.size * 0.5);
          this.p.line(0, 0, -this.size * 1.2, this.size * 0.8);
          this.p.noStroke();

          // 翅膀上的梦幻斑点 (🦋 标志性特征)
          this.p.fill(255, this.life * 0.6);
          this.p.ellipse(-this.size * 1.2, -this.size * 0.4, this.size * 0.3, this.size * 0.2);
          this.p.ellipse(-this.size * 0.8, this.size * 0.8, this.size * 0.2, this.size * 0.2);
          
          this.p.pop();
        }

        // 身体 (简化为一个点)
        this.p.fill(20, 20, 20, this.life);
        this.p.ellipse(0, 0, this.size * 0.1, this.size * 0.1);
        
        // 触角 (弯曲更自然)
        this.p.noFill();
        this.p.stroke(255, this.life * 0.7);
        this.p.strokeWeight(1.5);
        this.p.beginShape();
        this.p.vertex(-2, -this.size * 0.4);
        this.p.bezierVertex(-5, -this.size * 0.6, -8, -this.size * 0.8, -10, -this.size * 0.7);
        this.p.endShape();
        this.p.beginShape();
        this.p.vertex(2, -this.size * 0.4);
        this.p.bezierVertex(5, -this.size * 0.6, 8, -this.size * 0.8, 10, -this.size * 0.7);
        this.p.endShape();
        
        this.p.pop();
      }

      isDead() {
        return this.life <= 0;
      }
    }

    const p5Instance = new p5(sketch, containerRef.current);

    return () => {
      p5Instance.remove();
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="relative w-screen h-screen bg-black overflow-hidden font-sans">
      {/* 隐藏的视频元素用于 MediaPipe 处理 */}
      <video
        ref={videoRef}
        className="hidden"
        playsInline
        muted
      />

      {/* p5.js 画布容器 */}
      <div ref={containerRef} className="absolute inset-0 z-10" />

      {/* UI 覆盖层 */}
      <div className="absolute top-8 left-8 z-20 pointer-events-none flex flex-col gap-6">
        <div>
          <h1 className="text-4xl font-bold text-white tracking-tighter uppercase mb-2">
            Magic Cup
          </h1>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isLoading ? 'bg-yellow-500 animate-pulse' : 'bg-green-500'}`} />
            <p className="text-xs text-white/50 uppercase tracking-widest font-mono">
              {isLoading ? 'Initializing Detector...' : 'System Ready'}
            </p>
          </div>
        </div>

        {!isLoading && (
          <div className="bg-white/5 backdrop-blur-sm border border-white/5 rounded-xl p-3 max-w-fit">
            <h3 className="text-[9px] font-bold mb-2 tracking-[0.1em] uppercase opacity-30 text-white">
              魔法手势指南
            </h3>
            <ul className="space-y-1 text-[10px] text-white/60">
              <li className="flex items-center gap-2">
                <span className="w-1 h-1 rounded-full bg-yellow-400/60"></span>
                <span>完全张开五指：释放金色流星</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1 h-1 rounded-full bg-blue-400/60"></span>
                <span>双手合拢再打开：释放华丽蝴蝶</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1 h-1 rounded-full bg-purple-400/60"></span>
                <span>手指捏合：种下梦幻花朵</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1 h-1 rounded-full bg-pink-400/60"></span>
                <span>放置杯子：开出繁花</span>
              </li>
            </ul>
          </div>
        )}
      </div>

      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center z-30 bg-black">
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-yellow-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-yellow-500 font-mono text-sm tracking-widest uppercase">Loading Object Detector...</p>
          </div>
        </div>
      )}

      {error && (
        <div className="absolute inset-0 flex items-center justify-center z-40 bg-black/80 backdrop-blur-md">
          <div className="bg-red-500/10 border border-red-500 p-8 rounded-2xl max-w-md text-center">
            <p className="text-red-500 font-bold mb-4">{error}</p>
            <button 
              onClick={() => window.location.reload()}
              className="px-6 py-2 bg-red-500 text-white rounded-full text-sm font-bold hover:bg-red-600 transition-colors"
            >
              重试
            </button>
          </div>
        </div>
      )}

    </div>
  );
}
