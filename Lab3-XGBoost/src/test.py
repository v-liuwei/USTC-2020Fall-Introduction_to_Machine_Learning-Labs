class foo(object):
    a = 1

    @classmethod
    def set_a(x):
        foo.a = x

    def print_a(self):
        print(foo.a)


if __name__ == "__main__":
    foo.set_a(3)
    f = foo()
    g = foo()
    f.set_a(1)
    g.print_a()
