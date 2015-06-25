import os
from unittest import TestCase, TestLoader, main

from vsc.mympingpong.mympi import getshared


class MyMPITest(TestCase):

    """ Tests for mympi.py """

    def setUp(self):
        self.shared = os.environ['VSC_SCRATCH']

    def tearDown(self):
        os.environ['VSC_SCRATCH'] = self.shared

    def test_getshared(self):
        os.environ['VSC_SCRATCH'] = "/tmp"
        self.assertEqual(getshared(), "/tmp")

        del os.environ['VSC_SCRATCH']
        with self.assertRaises(KeyError):
            getshared()


def suite():
    """ returns all the testcases in this module """
    return TestLoader().loadTestsFromTestCase(MyMpiTest)


if __name__ == '__main__':
    main()
