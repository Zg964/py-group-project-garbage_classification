const path = require('path');
const pptxgen = require('pptxgenjs');
const html2pptx = require('C:/Users/zgzz1/.claude/skills/pptx/scripts/html2pptx');

async function createPresentation() {
    const pptx = new pptxgen();
    pptx.layout = 'LAYOUT_16x9';
    pptx.author = 'Garbage Classification Team';
    pptx.title = '智能垃圾分类系统';
    pptx.subject = '项目汇报';

    const slidesDir = 'C:/Users/zgzz1/Desktop/python-程序设计/py-group-project-garbage_classification/ppt_slides';
    const slides = ['slide1_title','slide2_background','slide3_data','slide4_augment',
                    'slide5_models','slide6_training','slide7_results1','slide8_results2','slide9_summary'];

    for (const s of slides) {
        console.log('Creating slide: ' + s);
        await html2pptx(path.join(slidesDir, s + '.html'), pptx);
    }

    const outputPath = 'C:/Users/zgzz1/Desktop/python-程序设计/py-group-project-garbage_classification/项目汇报PPT.pptx';
    await pptx.writeFile({ fileName: outputPath });
    console.log('Saved: ' + outputPath);
}

createPresentation().catch(err => { console.error('Error:', err.message); process.exit(1); });
