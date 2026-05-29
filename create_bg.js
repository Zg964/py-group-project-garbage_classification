const sharp = require('sharp');

async function createGradientBackground(filename, c1, c2) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="810">
    <defs>
      <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:${c1}"/>
        <stop offset="100%" style="stop-color:${c2}"/>
      </linearGradient>
    </defs>
    <rect width="100%" height="100%" fill="url(#g)"/>
  </svg>`;
  await sharp(Buffer.from(svg)).png().toFile(filename);
  console.log('Created ' + filename);
}

async function createAccentBar() {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="8" height="120">
    <rect width="8" height="120" fill="#1565C0"/>
  </svg>`;
  await sharp(Buffer.from(svg)).png().toFile('ppt_slides/accent_bar.png');
  console.log('Created accent bar');
}

async function main() {
  await createGradientBackground('ppt_slides/bg_title.png', '#0D47A1', '#1565C0');
  await createGradientBackground('ppt_slides/bg_section.png', '#1565C0', '#1976D2');
  await createAccentBar();
}
main().catch(console.error);
