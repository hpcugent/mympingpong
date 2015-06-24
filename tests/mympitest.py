import unittest,os

from vsc.mympingpong.mympi import getshared

class mympitest(unittest.TestCase):

    def setUp(self):
        self.shared = os.environ['VSC_SCRATCH']

    def tearDown(self):
        os.environ['VSC_SCRATCH'] = self.shared

    def test_getshared(self):

        os.environ['VSC_SCRATCH'] = "/tmp"
        self.assertEqual(getshared(),"/tmp")

        del os.environ['VSC_SCRATCH']
        with self.assertRaises(KeyError):
            getshared()

if __name__ == '__main__':
    unittest.main()
