"""Small real Pillow fixtures for publication decoding order; no run data."""
import binascii
from pathlib import Path
import struct
import tempfile
import unittest

from PIL import Image

from .public_pack import inspect_public_image


class PublicImageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def save(self, filename, *, exif=False):
        path = self.root / filename
        options = {}
        if exif:
            tags = Image.Exif(); tags[315] = 'SYNTHETIC_FIXTURE_EXIF'
            options['exif'] = tags
        Image.new('RGB', (7, 5), (20, 60, 100)).save(path, **options)
        return path

    def test_valid_png_verifies_before_metadata_and_loads_pixels(self):
        path = self.save('valid.png'); problems = []
        self.assertEqual(inspect_public_image(path, path.name, problems), [7, 5])
        self.assertEqual(problems, [])

    def test_valid_jpeg_is_decoded_without_false_exif_flag(self):
        path = self.save('valid.jpg'); problems = []
        self.assertEqual(inspect_public_image(path, path.name, problems), [7, 5])
        self.assertEqual(problems, [])

    def test_png_exif_remains_a_publication_problem(self):
        path = self.save('private_metadata.png', exif=True); problems = []
        self.assertEqual(inspect_public_image(path, path.name, problems), [7, 5])
        self.assertEqual(problems, [dict(kind='IMAGE_EXIF', path=path.name)])

    def test_jpeg_exif_remains_a_publication_problem(self):
        path = self.save('private_metadata.jpg', exif=True); problems = []
        self.assertEqual(inspect_public_image(path, path.name, problems), [7, 5])
        self.assertEqual(problems, [dict(kind='IMAGE_EXIF', path=path.name)])

    def test_bad_png_crc_is_rejected(self):
        path = self.save('corrupt_crc.png'); data = bytearray(path.read_bytes())
        start = data.index(b'IDAT'); data[start + 4] ^= 0xff
        path.write_bytes(data)
        with self.assertRaises(Exception):
            inspect_public_image(path, path.name, [])

    def test_valid_crc_with_broken_pixel_stream_is_still_rejected(self):
        path = self.save('corrupt_pixels.png'); data = bytearray(path.read_bytes())
        start = data.index(b'IDAT'); size = struct.unpack('>I', data[start-4:start])[0]
        data[start+4] = 0  # invalid zlib header; repair CRC to exercise load(), not verify()
        checksum = binascii.crc32(data[start:start+4+size]) & 0xffffffff
        data[start+4+size:start+8+size] = struct.pack('>I', checksum)
        path.write_bytes(data)
        with Image.open(path) as image:
            image.verify()  # structural/chunk validation passes this constructed input
        with self.assertRaises(Exception):
            inspect_public_image(path, path.name, [])

    def test_truncated_jpeg_is_rejected(self):
        path = self.save('truncated.jpg'); data = path.read_bytes(); path.write_bytes(data[:len(data)//2])
        with self.assertRaises(Exception):
            inspect_public_image(path, path.name, [])


if __name__ == '__main__':
    unittest.main()
