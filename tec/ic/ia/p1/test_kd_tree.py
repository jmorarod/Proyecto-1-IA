import pytest
from kd_tree import *


def test_distance():
    result = distance([4, 5, 6, 7, 8, 6], [1, 2, 8, 6, 4, 8])
    assert result == 6.244997998398398


def test_on_non_list_arguments_distance():
    with pytest.raises(TypeError):
        distance("abc", [1, 2, 3])


def test_closest_point():
    result = closest_point([[4, 5, 6, 7, 8, 6], [1, 2, 8, 6, 4, 8],
                            [8, 1, 4, 6, 8, 9]], [1, 2, 7, 8, 9, 1])
    assert result == [4, 5, 6, 7, 8, 6]


def test_on_non_type_arguments_closest_point():
    with pytest.raises(TypeError):
        closest_point([4, 5, 7], [1, 2, 3])


def test_create_csv():
    result = create_csv([["Campo1", "Campo2"], [[0, 1, 2, 3, 4], 0, 1]])
    assert result is None


def test_on_non_type_parameter_create_csv():
    with pytest.raises(TypeError):
        create_csv([[0, 1], "Campo"])


def test_on_non_type_parameter_list_create_csv():
    with pytest.raises(TypeError):
        create_csv([[0, 1], [1, 2]])


def test_on_non_proper_len_parameter_create_csv():
    with pytest.raises(AttributeError):
        create_csv([["Campo1", "Campo2"], [[0, 1, 2, 1], 0, 1]])


def test_insert_node_right():
    root = Node(0, [0, 1])
    result = root.insert_node([2, 0], 1)
    assert result == root.right


def test_insert_node_left():
    root = Node(0, [0, 1])
    result = root.insert_node([-1, 0], 1)
    assert result == root.left


def test_on_non_type_parameter_insert_node():
    root = Node(0, [0, 1])
    with pytest.raises(TypeError):
        root.insert_node(1, "Node")


def test_on_non_type_parameter_node():
    root = Node(0, [0, 1])
    with pytest.raises(TypeError):
        root.insert_node(1, "Node")


def test_insert_leaf():
    root = Node(0, [0, 1])
    result = root.insert_leaf([1, 2], True)
    assert result is None


def test_on_non_type_parameter_insert_leaf():
    root = Node(0, [0, 1])
    with pytest.raises(TypeError):
        root.insert_leaf([1, 2], 4)


def test_create_kd_tree():
    result = create_kd_tree([[1, 2], [2, 3]], 2)
    assert isinstance(result, Node)


def test_create_kd_tree_aux():
    result = create_kd_tree_aux([[1, 2], [2, 3]], 2, 2, None, 0)
    assert isinstance(result, Node)


def test_closer_distance():
    left = Node(0, [1, 2, 3])
    right = Node(0, [5, 8, 4])
    result = closer_distance([7, 1, 2], left, right)
    assert result == left


def test_kd_tree_find_neighbors():
    data = [[7, 8, 9, 4, 5, 6], [1, 2, 3, 7, 5, 6], [5, 7, 8, 3, 1, 9],
            [10, 12, 4, 6, 4, 7], [4, 9, 4, 7, 1, 3], [4, 7, 13, 9, 1, 2]]
    root = create_kd_tree(data, 2)
    test = [4, 5, 6, 7, 8, 6]
    result = kd_tree_find_neighbors(root, test)
    assert result == 6


def test_get_nearest_neighbors():
    data = [[7, 8, 9, 4, 5, 7], [1, 2, 3, 7, 5, 6], [5, 7, 8, 3, 1, 9],
            [10, 12, 4, 6, 4, 7], [4, 9, 4, 7, 1, 1], [4, 7, 13, 9, 1, 2]]
    result = get_nearest_neighbors(data)
    assert result == 7


def test_get_nearest_neighbors_aux():
    data = {1: 8, 2: 5, 4: 4, 6: 9}
    result = get_nearest_neighbors_aux(data)
    assert result == 6


def test_kd_tree():
    result = kd_tree(10, 2, 20)
    assert result is None


def test_kd_tree_aux():
    myData = [['poblacion_canton', 'superficie_canton', 'densidad_poblacion',
               'urbano', 'sexo', 'dependencia_demografica', 'ocupa_vivienda',
               'promedio_ocupantes', 'vivienda_buen_estado',
               'vivienda_hacinada', 'alfabetismo', 'escolaridad_promedio',
               'educacion_regular', 'fuera_fuerza_trabajo',
               'participacion_fuerza_trabajo', 'asegurado', 'extranjero',
               'discapacidad', 'no_asegurado', 'porcentaje_jefatura_femenina',
               'porcentaje_jefatura_compartida', 'edad', 'voto_primera_ronda',
               'voto_segunda_ronda', 'es_entrenamiento', 'prediccion_r1',
               'prediccion_r2', 'prediccion_r2_con_r1']]
    muestra = generar_muestra_pais(10)
    data = datos_r1_normalizados(muestra)
    data = np.array(data).tolist()
    result = kd_tree_aux(data, 2, 2, myData, 0)
    assert len(result) == 11
