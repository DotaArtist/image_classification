#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'


def yield_test(list_a):
    for i in range(len(list_a)):
        print(i)
        yield list_a[i]


if __name__ == '__main__':
    a = yield_test(["a", "b", "c", "d"])
    for i in a:
        print(i)
