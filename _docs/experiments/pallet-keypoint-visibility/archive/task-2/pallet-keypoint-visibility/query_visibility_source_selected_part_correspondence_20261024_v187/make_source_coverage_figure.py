"""Static source-only figure from the sealed descriptive count table."""
from pathlib import Path
import hashlib
import json
from PIL import Image, ImageDraw, ImageFont

V = Path(__file__).absolute().parent
SOURCE = V / 'descriptive_source_tables_v1/SUMMARY.json'
OUT = V / 'source_coverage_figure_v1'


def main():
    summary = json.loads(SOURCE.read_text())
    assert summary['PASS'] and summary['all_queries']['total'] == 1480 and not summary['fit_admission']
    OUT.mkdir(exist_ok=False)
    im = Image.new('RGB', (1320, 760), '#101827')
    draw = ImageDraw.Draw(im)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    title = ImageFont.truetype(font_path, 27)
    font = ImageFont.truetype(font_path, 20)
    small = ImageFont.truetype(font_path, 17)
    colors = ['#49c9ae', '#f4a261', '#8e9bb5', '#465169']
    labels = ['Available + nominal ref inside crop', 'Available + nominal ref outside crop',
              'Available + obj1 association UNKNOWN', 'Original input U']
    draw.text((32, 24), 'SOURCE TRAIN: original crop / nominal part correspondence', font=title, fill='white')
    draw.text((32, 67), '185 frames x 8 native slots = 1,480 rows. Visibility D/E/S stays UNKNOWN.', font=font, fill='#d1d9e6')
    totals = [0, 0, 0, 0]
    for k in range(8):
        r = summary['per_native'][str(k)]
        values = [r['eligible_inside_original_crop_extent'], r['eligible_outside_original_crop_extent'],
                  r['input_available']-r['eligible_available_weak_nominal'], r['inputU']]
        assert sum(values) == 185
        y = 141 + k * 53
        draw.text((32, y+7), 'native ' + str(k), font=font, fill='white')
        x = 157
        for j, value in enumerate(values):
            width = value * 5.4
            if value:
                draw.rectangle((x, y, x+width, y+35), fill=colors[j])
                if width > 25:
                    text = str(value)
                    draw.text((x+width/2-draw.textlength(text, font=small)/2, y+7), text, font=small, fill='#07101d' if j < 2 else 'white')
            x += width
            totals[j] += value
        draw.text((1171, y+7), '185', font=small, fill='#d1d9e6')
    assert totals == [650, 274, 362, 194]
    for j, label in enumerate(labels):
        x = 32 + (j % 2) * 650
        y = 601 + (j // 2) * 42
        draw.rectangle((x, y+4, x+22, y+26), fill=colors[j])
        draw.text((x+33, y), f'{label}: {totals[j]}', font=small, fill='white')
    draw.text((32, 704), 'Nominal cuboid + provided pose; neither physical endpoint gold nor visibility labels.', font=small, fill='#bac8dc')
    path = OUT / 'PUBLIC185_NATIVE8_CROP_COVERAGE.png'
    im.save(path)
    result = {'completed': True, 'PASS': True, 'source': {'path': str(SOURCE), 'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
              'code': {'path': str(Path(__file__).absolute()), 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              'image': {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
              'counts': totals, 'original_input_pixel_reads': 0, 'fit_admission': False}
    with (OUT / 'COMPLETE.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
