from pyzonotope.Zonotope import Zonotope
import numpy as np

def test1():
    a = Zonotope([0.5, 0.5])
    #print(type(a.Z), a.cont_set._id, a.cont_set._dimension)
    np.testing.assert_array_equal(a.Z, np.array([0.5, 0.5]))
    assert a.cont_set._id == 0
    assert a.cont_set._dimension == 1
    print("test 1 Succedded")

def test2():
    a = Zonotope(0.5, 0.5)
    #print(type(a.Z), a.cont_set._id, a.cont_set._dimension)
    np.testing.assert_array_equal(a.Z, np.array([0.5, 0.5]))
    assert a.cont_set._id == 0
    assert a.cont_set._dimension == 1
    print("Test 2 Succedded")

def test3():
    a = Zonotope(np.array(np.ones((3, 1))), 0.1 * np.diag(np.ones((3, 1)).T[0]))
    target_z = np.array([[1, 0.1, 0, 0], [1, 0, 0.1, 0], [1, 0, 0, 0.1]])

    np.testing.assert_array_equal(a.Z, target_z)
    np.testing.assert_array_equal(a.center(), np.ones((3, 1)))
    np.testing.assert_array_equal(a.generators(), 0.1 * np.diag(np.ones((3, 1)).T[0]))

    assert a.cont_set._id == 0
    assert a.cont_set._dimension == 3
    
    print("Test 3 Succedded")


test1()
test2()
test3()