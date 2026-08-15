/**
 * Matrix ASCII Rain Canvas Background.
 * Renders glowing matrix glyphs falling gently in the terminal background.
 */
export function initMatrixRain(canvasId = 'matrix-canvas') {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let animationId;
  const glyphs = 'SETHアイウエオカキクケコサシスセソ01ΩΔΣ0123456789';

  let cols = 0;
  let drops = [];

  const setup = () => {
    canvas.width = canvas.parentElement.offsetWidth || window.innerWidth;
    canvas.height = canvas.parentElement.offsetHeight || window.innerHeight;
    cols = Math.floor(canvas.width / 16);
    drops = new Array(cols).fill(0);
  };

  setup();
  window.addEventListener('resize', setup);

  const draw = () => {
    ctx.fillStyle = 'rgba(3, 3, 8, 0.12)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#00f0ff';
    ctx.font = '14px monospace';

    for (let i = 0; i < drops.length; i++) {
      const char = glyphs[Math.floor(Math.random() * glyphs.length)];
      ctx.fillText(char, i * 16, drops[i] * 16);
      if (drops[i] * 16 > canvas.height && Math.random() > 0.975) {
        drops[i] = 0;
      }
      drops[i]++;
    }
    animationId = requestAnimationFrame(draw);
  };

  draw();

  return () => {
    cancelAnimationFrame(animationId);
    window.removeEventListener('resize', setup);
  };
}
