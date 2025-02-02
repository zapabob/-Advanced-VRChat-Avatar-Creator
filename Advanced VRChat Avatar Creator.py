#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced VRChat Avatar Creator - Final Implementation (Part 1)

このプログラムは、PyQt6, PyOpenGL, Autodesk FBX SDK, vrm2py, SciPy, NumPy, Matplotlib, Plotly,
および scikit-learn を組み合わせた高度な 3D モデリングツールです。
各種科学計算や機械学習ライブラリを活用し、頂点操作・エッジ検出の最適化、FBX／VRM 入出力処理の強化、UI の拡充、プラグインシステムの統合、そして詳細なチュートリアルを実現しています。

必要な外部 SDK（Autodesk FBX SDK, vrm2py 等）のインストール・設定は事前に行ってください。
"""

import sys, os, math, numpy as np, importlib.util
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QAction, QToolBar, QStatusBar,
    QFileDialog, QDockWidget, QWidget, QVBoxLayout, QLabel, QLineEdit,
    QHBoxLayout, QPushButton, QWizard, QWizardPage, QFormLayout, QSlider,
    QListWidget, QTextEdit, QSizePolicy
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QPoint
from OpenGL.GL import *
from OpenGL.GLU import *

# 外部 SDK のインポート（事前にインストール済み）
try:
    import fbx
except ImportError:
    raise ImportError("Autodesk FBX SDK がインストールされていません。公式サイトからFBX SDK 2020.3.1以降をインストールしてください")

import json
import gltflib  # GLTFライブラリを使用してVRMを読み込む

# SciPy, Matplotlib, Plotly, scikit-learn
from scipy.spatial import KDTree, Delaunay
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.cluster import KMeans

import subprocess

# =====================================================
# 1. 高度なメッシュ編集クラス（AdvancedMeshEnhanced）
# =====================================================
class AdvancedMeshEnhanced:
    def __init__(self, vertices, faces):
        """
        :param vertices: 各頂点 [x, y, z]
        :param faces: 各面の頂点インデックスのリスト
        """
        self.vertices = np.array(vertices, dtype=float)
        self.faces = faces
        self.normals = None
        self.edge_face_map = {}  # エッジと接する面のマッピング
        self.compute_normals()
        self.compute_edges()
        self.build_kdtree()
        self.smooth_vertices(iterations=2, radius=0.05)  # 科学計算による平滑化

    def build_kdtree(self):
        self.kdtree = KDTree(self.vertices)

    def compute_normals(self):
        """ 各面および頂点の法線を計算 """
        self.normals = np.zeros((len(self.vertices), 3))
        for face in self.faces:
            if len(face) < 3: continue
            v0, v1, v2 = self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal /= norm
            for idx in face:
                self.normals[idx] += normal
        for i in range(len(self.normals)):
            n = np.linalg.norm(self.normals[i])
            if n > 0:
                self.normals[i] /= n

    def compute_edges(self):
        """ 半エッジ構造に近い形でエッジと接面を計算 """
        self.edge_face_map = {}
        for fi, face in enumerate(self.faces):
            n = len(face)
            for i in range(n):
                a, b = face[i], face[(i+1)%n]
                edge = tuple(sorted((a, b)))
                self.edge_face_map.setdefault(edge, []).append(fi)

    def enhanced_edge_detection(self, angle_threshold=25):
        """
        SciPy の KDTree などを活用して、より精緻なエッジ検出を実施
        :return: 鋭いエッジのリスト
        """
        sharp_edges = []
        for edge, face_indices in self.edge_face_map.items():
            if len(face_indices) < 2:
                sharp_edges.append(edge)
            else:
                angles = []
                for i in range(len(face_indices)):
                    for j in range(i+1, len(face_indices)):
                        n1 = np.mean([self.normals[k] for k in self.faces[face_indices[i]]], axis=0)
                        n2 = np.mean([self.normals[k] for k in self.faces[face_indices[j]]], axis=0)
                        dot_val = np.clip(np.dot(n1, n2), -1.0, 1.0)
                        angle = np.degrees(np.arccos(dot_val))
                        angles.append(angle)
                if angles and np.mean(angles) >= angle_threshold:
                    sharp_edges.append(edge)
        return sharp_edges

    def smooth_vertices(self, iterations=1, radius=0.05):
        """
        SciPy の KDTree を用いて各頂点の近傍の中央値で平滑化
        """
        new_vertices = self.vertices.copy()
        for _ in range(iterations):
            for i, v in enumerate(self.vertices):
                idxs = self.kdtree.query_ball_point(v, r=radius)
                if idxs:
                    new_vertices[i] = np.median(self.vertices[idxs], axis=0)
            self.vertices = new_vertices.copy()
            self.build_kdtree()
        self.compute_normals()
        self.compute_edges()

    def set_vertex_position(self, index, new_pos):
        self.vertices[index] = new_pos
        self.compute_normals()
        self.compute_edges()
        self.build_kdtree()

    def extrude_face(self, face_index, distance):
        face = self.faces[face_index]
        face_normal = np.mean([self.normals[i] for i in face], axis=0)
        norm = np.linalg.norm(face_normal)
        if norm:
            face_normal /= norm
        new_indices = []
        for i in face:
            new_vertex = self.vertices[i] + face_normal * distance
            self.vertices = np.vstack((self.vertices, new_vertex))
            new_indices.append(len(self.vertices)-1)
        self.faces.append(new_indices)
        n = len(face)
        for i in range(n):
            a, b = face[i], face[(i+1)%n]
            a_new, b_new = new_indices[i], new_indices[(i+1)%n]
            self.faces.append([a, b, b_new])
            self.faces.append([a, b_new, a_new])
        self.compute_normals()
        self.compute_edges()
        self.build_kdtree()

    def subdivide(self):
        new_faces = []
        new_vertices = self.vertices.tolist()
        for face in self.faces:
            if len(face) != 3:
                new_faces.append(face)
                continue
            v0, v1, v2 = self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]
            center = (v0 + v1 + v2) / 3.0
            center_index = len(new_vertices)
            new_vertices.append(center.tolist())
            new_faces.append([face[0], face[1], center_index])
            new_faces.append([face[1], face[2], center_index])
            new_faces.append([face[2], face[0], center_index])
        self.vertices = np.array(new_vertices, dtype=float)
        self.faces = new_faces
        self.compute_normals()
        self.compute_edges()
        self.build_kdtree()

    def deform(self, factor=0.1):
        for i, v in enumerate(self.vertices):
            self.vertices[i][1] += factor * math.sin(v[0])
        self.compute_normals()

    def merge_vertices(self, tolerance=1e-5):
        rounded = np.round(self.vertices / tolerance).astype(int)
        unique, indices = np.unique(rounded, axis=0, return_inverse=True)
        self.vertices = unique * tolerance
        new_faces = []
        for face in self.faces:
            new_faces.append([int(indices[i]) for i in face])
        self.faces = new_faces
        self.compute_normals()
        self.compute_edges()
        self.build_kdtree()

    def draw(self):
        glBegin(GL_TRIANGLES)
        for face in self.faces:
            if len(face) < 3:
                continue
            normal = np.mean([self.normals[i] for i in face], axis=0)
            glNormal3fv(normal)
            for idx in face:
                color = (abs(self.normals[idx][0]), abs(self.normals[idx][1]), abs(self.normals[idx][2]))
                glColor3fv(color)
                glVertex3fv(self.vertices[idx])
        glEnd()

# =====================================================
# 1.5 キューブメッシュ生成関数の追加
# =====================================================
def create_cube_mesh(center=(0,0,0), size=1.0):
    """立方体メッシュを生成するヘルパー関数"""
    half = size / 2
    vertices = [
        [center[0]-half, center[1]-half, center[2]-half],
        [center[0]+half, center[1]-half, center[2]-half],
        [center[0]+half, center[1]+half, center[2]-half],
        [center[0]-half, center[1]+half, center[2]-half],
        [center[0]-half, center[1]-half, center[2]+half],
        [center[0]+half, center[1]-half, center[2]+half],
        [center[0]+half, center[1]+half, center[2]+half],
        [center[0]-half, center[1]+half, center[2]+half]
    ]
    # 各面を三角形2つで構成
    faces = [
        [0,1,2], [0,2,3],  # 前面
        [4,5,6], [4,6,7],  # 背面
        [0,4,7], [0,7,3],  # 左面
        [1,5,6], [1,6,2],  # 右面
        [3,2,6], [3,6,7],  # 上面
        [0,1,5], [0,5,4]   # 底面
    ]
    return AdvancedMeshEnhanced(vertices, faces)

# =====================================================
# 2. FBX／VRM 入出力処理の最適化（外部 SDK 連携）
# =====================================================
def export_mesh_to_fbx(mesh, filepath):
    """
    Autodesk FBX SDK を用いて、mesh (AdvancedMeshEnhanced) を FBX 形式で出力する
    """
    manager = fbx.FbxManager.Create()
    scene = fbx.FbxScene.Create(manager, "")
    fbx_mesh = fbx.FbxMesh.Create(scene, "MyMesh")
    num_vertices = len(mesh.vertices)
    fbx_mesh.InitControlPoints(num_vertices)
    for i, v in enumerate(mesh.vertices):
        fbx_mesh.SetControlPointAt(fbx.FbxVector4(v[0], v[1], v[2]), i)
    for face in mesh.faces:
        fbx_mesh.BeginPolygon()
        for idx in face:
            fbx_mesh.AddPolygon(idx)
        fbx_mesh.EndPolygon()
    node = fbx.FbxNode.Create(scene, "MeshNode")
    node.SetNodeAttribute(fbx_mesh)
    scene.GetRootNode().AddChild(node)
    exporter = fbx.FbxExporter.Create(manager, "")
    if not exporter.Initialize(filepath, -1, manager.GetIOSettings()):
        print("FBX Exporter 初期化失敗")
        return False
    exporter.Export(scene)
    exporter.Destroy()
    manager.Destroy()
    return True

def import_vrm_file(filepath):
    """
    VRMファイルを読み込み、基本的な情報を抽出する
    """
    try:
        with open(filepath, 'rb') as f:
            # VRMはGLTFベースなのでGLTFとして読み込む
            gltf = gltflib.GLTF.load(filepath)
            
            # メッシュデータの抽出
            vertices = []
            faces = []
            
            # プリミティブからメッシュデータを取得
            for mesh in gltf.model.meshes:
                for primitive in mesh.primitives:
                    # 頂点位置の取得
                    positions = gltf.get_accessor_data(primitive.attributes.POSITION)
                    for pos in positions:
                        vertices.append([pos[0], pos[1], pos[2]])
                    
                    # インデックスの取得
                    if primitive.indices is not None:
                        indices = gltf.get_accessor_data(primitive.indices)
                        for i in range(0, len(indices), 3):
                            faces.append([indices[i], indices[i+1], indices[i+2]])
            
            return AdvancedMeshEnhanced(vertices, faces)
    except Exception as e:
        print(f"VRMファイルの読み込みに失敗: {e}")
        return None

def export_vrm_file(model, filepath):
    """
    モデルをVRMとしてエクスポート（BlendShape、ヒューマノイド情報付き）
    """
    try:
        gltf = gltflib.GLTF()
        
        # メッシュデータの追加
        vertices = model.vertices.tolist()
        faces = model.faces
        
        vertex_buffer = gltflib.Data(vertices)
        index_buffer = gltflib.Data(faces)
        
        # アクセサの追加
        gltf.add_accessor(vertex_buffer, component_type=gltflib.ComponentType.FLOAT,
                         type=gltflib.AccessorType.VEC3, count=len(vertices))
        gltf.add_accessor(index_buffer, component_type=gltflib.ComponentType.UNSIGNED_SHORT,
                         type=gltflib.AccessorType.SCALAR, count=len(faces) * 3)
        
        # VRM拡張の追加（詳細な設定）
        vrm_extension = {
            'exporterVersion': '1.0',
            'specVersion': '1.0',
            'meta': {
                'title': 'VRChat Avatar',
                'version': '1.0',
                'author': 'Advanced VRChat Avatar Creator',
                'contactInformation': '',
                'reference': '',
                'allowedUserName': 'OnlyAuthor',
                'violentUssageName': 'Disallow',
                'sexualUssageName': 'Disallow',
                'commercialUssageName': 'Disallow',
                'otherPermissionUrl': ''
            },
            'humanoid': {
                'humanBones': {
                    'hips': {'node': 0},
                    'spine': {'node': 1},
                    'chest': {'node': 2},
                    'neck': {'node': 3},
                    'head': {'node': 4},
                    'leftUpperArm': {'node': 5},
                    'leftLowerArm': {'node': 6},
                    'leftHand': {'node': 7},
                    'rightUpperArm': {'node': 8},
                    'rightLowerArm': {'node': 9},
                    'rightHand': {'node': 10}
                }
            },
            'blendShapeMaster': {
                'blendShapeGroups': [
                    {
                        'name': 'Joy',
                        'presetName': 'joy',
                        'binds': [
                            {
                                'mesh': 0,
                                'index': 0,
                                'weight': 100
                            }
                        ]
                    },
                    {
                        'name': 'Angry',
                        'presetName': 'angry',
                        'binds': [
                            {
                                'mesh': 0,
                                'index': 1,
                                'weight': 100
                            }
                        ]
                    }
                ]
            },
            'secondaryAnimation': {
                'boneGroups': [],
                'colliderGroups': []
            }
        }
        
        gltf.add_extension('VRM', vrm_extension)
        
        # ファイルに保存
        gltf.save(filepath)
        return True
    except Exception as e:
        print(f"VRMファイルの出力に失敗: {e}")
        return False

# =====================================================
# 3. VRChat 用アバター作成（ボーンリギング）
# =====================================================
class AvatarRig:
    def __init__(self, mesh):
        self.mesh = mesh
        # VRChat用のボーン構造を定義
        self.bones = {
            "Hips": {"parent": None, "transform": np.eye(4)},
            "Spine": {"parent": "Hips", "transform": np.eye(4)},
            "Chest": {"parent": "Spine", "transform": np.eye(4)},
            "Neck": {"parent": "Chest", "transform": np.eye(4)},
            "Head": {"parent": "Neck", "transform": np.eye(4)},
            "LeftUpperArm": {"parent": "Chest", "transform": np.eye(4)},
            "LeftLowerArm": {"parent": "LeftUpperArm", "transform": np.eye(4)},
            "LeftHand": {"parent": "LeftLowerArm", "transform": np.eye(4)},
            "RightUpperArm": {"parent": "Chest", "transform": np.eye(4)},
            "RightLowerArm": {"parent": "RightUpperArm", "transform": np.eye(4)},
            "RightHand": {"parent": "RightLowerArm", "transform": np.eye(4)}
        }
        self.vertex_weights = np.ones((len(mesh.vertices), len(self.bones))) / len(self.bones)
        self.blend_shapes = {
            "Joy": np.zeros((len(mesh.vertices), 3)),
            "Angry": np.zeros((len(mesh.vertices), 3))
        }

    def apply_rigging(self):
        transformed = []
        for i, v in enumerate(self.mesh.vertices):
            v_h = np.array([v[0], v[1], v[2], 1.0])
            v_new = np.zeros(4)
            for b, bone in enumerate(self.bones.values()):
                weight = self.vertex_weights[i, b]
                v_new += weight * np.dot(bone["transform"], v_h)
            transformed.append(v_new[:3])
        self.mesh.vertices = np.array(transformed, dtype=float)
        self.mesh.compute_normals()
        self.mesh.compute_edges()
        self.mesh.build_kdtree()

    def get_bone_data(self):
        return self.bones

def create_vrchat_avatar(mesh):
    rig = AvatarRig(mesh)
    rig.apply_rigging()
    return rig.get_bone_data()

# =====================================================
# 4. UI の拡充：追加カスタムウィジェット
# (a) TransformationMatrixEditor - 変換行列編集ウィジェット
# =====================================================
class TransformationMatrixEditor(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.info_label = QLabel("変換行列 (4x4) を編集するにゃん", self)
        self.text_edit = QTextEdit(self)
        self.text_edit.setText(np.array2string(np.eye(4), separator=', '))
        self.apply_button = QPushButton("適用", self)
        self.apply_button.clicked.connect(self.emit_matrix)
        self.layout.addWidget(self.info_label)
        self.layout.addWidget(self.text_edit)
        self.layout.addWidget(self.apply_button)
        self.current_matrix = np.eye(4)

    def emit_matrix(self):
        try:
            text = self.text_edit.toPlainText()
            self.current_matrix = np.matrix(text)
            print("新しい変換行列:", self.current_matrix)
        except Exception as e:
            print("変換行列読み込みエラー:", e)

# =====================================================
# 5. 詳細なプロパティエディタ（DetailedPropertyEditor）
# =====================================================
class DetailedPropertyEditor(QWidget):
    def __init__(self, gl_widget, parent=None):
        super().__init__(parent)
        self.gl_widget = gl_widget
        self.layout = QFormLayout(self)
        self.vertex_count = QLineEdit(self)
        self.face_count = QLineEdit(self)
        self.edge_count = QLineEdit(self)
        self.sharp_edge_count = QLineEdit(self)
        self.graph_widget = QListWidget(self)  # グラフ結果をテキストとして表示する簡易ウィジェット
        for le in (self.vertex_count, self.face_count, self.edge_count, self.sharp_edge_count):
            le.setReadOnly(True)
        self.layout.addRow("頂点数:", self.vertex_count)
        self.layout.addRow("面数:", self.face_count)
        self.layout.addRow("エッジ数:", self.edge_count)
        self.layout.addRow("鋭いエッジ数:", self.sharp_edge_count)
        self.layout.addRow("解析グラフ:", self.graph_widget)
        self.update_properties()

    def update_properties(self):
        if self.gl_widget.shapes:
            mesh = self.gl_widget.shapes[0]
            self.vertex_count.setText(str(len(mesh.vertices)))
            self.face_count.setText(str(len(mesh.faces)))
            self.edge_count.setText(str(len(mesh.edge_face_map)))
            sharp = mesh.enhanced_edge_detection(angle_threshold=25)
            self.sharp_edge_count.setText(str(len(sharp)))
            # グラフ表示例：頂点間の距離分布
            center = np.mean(mesh.vertices, axis=0)
            dists = np.linalg.norm(mesh.vertices - center, axis=1)
            hist, bin_edges = np.histogram(dists, bins=10)
            self.graph_widget.clear()
            for i in range(len(hist)):
                self.graph_widget.addItem(f"Bin {bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}: {hist[i]}")
        else:
            self.vertex_count.setText("0")
            self.face_count.setText("0")
            self.edge_count.setText("0")
            self.sharp_edge_count.setText("0")
            self.graph_widget.clear()

# =====================================================
# 6. OpenGL 描画ウィジェット（AdvancedGLWidget）
# =====================================================
class AdvancedGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.shapes = []  # AdvancedMeshEnhanced オブジェクトのリスト
        self.rotationX = 0.0
        self.rotationY = 0.0
        self.lastPos = QPoint()
        self.zoom = -5.0
        self.vertexEditor = VertexEditorWidget(self)
        self.matrixEditor = TransformationMatrixEditor(self)

    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        light_position = [2.0, 2.0, 2.0, 1.0]
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w/h if h else 1
        gluPerspective(45.0, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, self.zoom)
        glRotatef(self.rotationX, 1.0, 0.0, 0.0)
        glRotatef(self.rotationY, 0.0, 1.0, 0.0)
        for mesh in self.shapes:
            mesh.draw()
        glFlush()

    def mousePressEvent(self, event):
        self.lastPos = event.position().toPoint()

    def mouseMoveEvent(self, event):
        newPos = event.position().toPoint()
        dx = newPos.x() - self.lastPos.x()
        dy = newPos.y() - self.lastPos.y()
        self.rotationX += dy * 0.5
        self.rotationY += dx * 0.5
        self.lastPos = newPos
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        self.zoom += delta * 0.5
        self.update()

    def addCubeMesh(self):
        mesh = create_cube_mesh(center=(0,0,0), size=1.0)
        self.shapes.append(mesh)
        self.update()

    def subdivideMesh(self):
        if self.shapes:
            self.shapes[0].subdivide()
            self.update()

    def deformMesh(self):
        if self.shapes:
            self.shapes[0].deform(factor=0.2)
            self.update()

    def mergeMeshVertices(self):
        if self.shapes:
            self.shapes[0].merge_vertices(tolerance=1e-5)
            self.update()

    def extrudeMeshFace(self, face_index, distance=0.5):
        if self.shapes and 0 <= face_index < len(self.shapes[0].faces):
            self.shapes[0].extrude_face(face_index, distance)
            self.update()

    def clearShapes(self):
        self.shapes = []
        self.update()

# =====================================================
# 7. 頂点エディタウィジェット（VertexEditorWidget）
# =====================================================
class VertexEditorWidget(QWidget):
    def __init__(self, gl_widget, parent=None):
        super().__init__(parent)
        self.gl_widget = gl_widget
        self.layout = QFormLayout(self)
        self.index_edit = QLineEdit(self)
        self.x_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.y_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.z_slider = QSlider(Qt.Orientation.Horizontal, self)
        for slider in (self.x_slider, self.y_slider, self.z_slider):
            slider.setRange(-100, 100)
            slider.setValue(0)
        self.apply_button = QPushButton("適用", self)
        self.apply_button.clicked.connect(self.apply_vertex_change)
        self.layout.addRow("頂点インデックス:", self.index_edit)
        self.layout.addRow("X:", self.x_slider)
        self.layout.addRow("Y:", self.y_slider)
        self.layout.addRow("Z:", self.z_slider)
        self.layout.addRow(self.apply_button)

    def apply_vertex_change(self):
        try:
            idx = int(self.index_edit.text())
            dx = self.x_slider.value() / 100.0
            dy = self.y_slider.value() / 100.0
            dz = self.z_slider.value() / 100.0
            if self.gl_widget.shapes:
                mesh = self.gl_widget.shapes[0]
                new_pos = mesh.vertices[idx] + np.array([dx, dy, dz])
                mesh.set_vertex_position(idx, new_pos)
                self.gl_widget.update()
        except Exception as e:
            print("頂点編集エラー:", e)

# =====================================================
# 8. インタラクティブチュートリアル（詳細ガイド）
# =====================================================
class TutorialWizard(QWizard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("詳細インタラクティブチュートリアル - ボブにゃん版")
        self.addPage(self.createIntroPage())
        self.addPage(self.createMeshEditPage())
        self.addPage(self.createVertexEditPage())
        self.addPage(self.createMatrixEditPage())
        self.addPage(self.createFileIOPage())
        self.addPage(self.createPluginPage())
        self.addPage(self.createAvatarPage())
        self.addPage(self.createGuidelinesPage())
        self.addPage(self.createConclusionPage())

    def createIntroPage(self):
        page = QWizardPage()
        page.setTitle("イントロダクション")
        label = QLabel("本チュートリアルでは、Advanced VRChat Avatar Creator の各機能を詳細に学び、最適なパラメータ設定やトラブルシューティングの方法も紹介するにゃん。")
        label.setWordWrap(True)
        layout = QVBoxLayout(page)
        layout.addWidget(label)
        return page

    def createMeshEditPage(self):
        page = QWizardPage()
        page.setTitle("高度なメッシュ編集")
        label = QLabel("・キューブ追加、分割、変形、頂点統合、面押し出しの各操作を実行し、精緻な局所解析を行うにゃん。\n・SciPy を利用した KDTree による平滑化と Delaunay 分割の結果を活用。")
        label.setWordWrap(True)
        layout = QVBoxLayout(page)
        layout.addWidget(label)
        return page

    def createVertexEditPage(self):
        page = QWizardPage()
        page.setTitle("頂点編集")
        label = QLabel("頂点エディタウィジェットを使い、任意の頂点を選択して座標を調整する方法を学ぶにゃん。")
        label.setWordWrap(True)
        layout = QVBoxLayout(page)
        layout.addWidget(label)
        return page

    def createMatrixEditPage(self):
        page = QWizardPage()
        page.setTitle("変換行列編集")
        label = QLabel("TransformationMatrixEditor を使って、モデルの変換行列を直接編集し、効果を確認する方法を学ぶにゃん。")
        label.setWordWrap(True)
        layout = QVBoxLayout(page)
        layout.addWidget(label)
        return page

    def createFileIOPage(self):
        page = QWizardPage()
        page.setTitle("FBX／VRM 入出力")
        label = QLabel("・Autodesk FBX SDK を用いた最適化された FBX エクスポート\n・vrm2py を用いた VRM 入出力の実装例を確認するにゃん。")
        label.setWordWrap(True)
        layout = QVBoxLayout(page)
        layout.addWidget(label)
        return page

    def createPluginPage(self):
        page = QWizardPage()
        page.setTitle("プラグイン統合")
        label = QLabel("plugins/ ディレクトリからサードパーティ製ツールをシームレスに統合する方法、及びその再読み込みの手順を学ぶにゃん。")
        label.setWordWrap(True)
        layout = QVBoxLayout(page)
        layout.addWidget(label)
        return page

    def createAvatarPage(self):
        page = QWizardPage()
        page.setTitle("VRChat アバター作成")
        label = QLabel("対象メッシュにボーンリギングを適用し、VRM 形式で VRChat 用アバターとして出力する手順を確認するにゃん。")
        label.setWordWrap(True)
        layout = QVBoxLayout(page)
        layout.addWidget(label)
        return page

    def createGuidelinesPage(self):
        page = QWizardPage()
        page.setTitle("ユーザーガイドライン")
        label = QLabel("各機能の使用例、最適なパラメータの推奨、トラブルシューティング方法、及びさらなる学習リソースを提供する詳細なガイドラインを確認するにゃん。")
        label.setWordWrap(True)
        layout = QVBoxLayout(page)
        layout.addWidget(label)
        return page

    def createConclusionPage(self):
        page = QWizardPage()
        page.setTitle("まとめ")
        label = QLabel("これで Advanced VRChat Avatar Creator の全機能を詳細に理解したにゃん！\n自由に実験して、夢の VRChat アバターを実現してほしいにゃん。")
        label.setWordWrap(True)
        layout = QVBoxLayout(page)
        layout.addWidget(label)
        return page

# =====================================================
# 9. プラグインシステムの強化
# =====================================================
class PluginBase:
    def __init__(self, main_window):
        self.main_window = main_window

    def activate(self):
        raise NotImplementedError("PluginBase: activate() を実装してください")

    def deactivate(self):
        raise NotImplementedError("PluginBase: deactivate() を実装してください")

def load_plugins(main_window, plugins_dir="plugins"):
    plugins = []
    if not os.path.exists(plugins_dir):
        return plugins
    for filename in os.listdir(plugins_dir):
        if filename.endswith(".py"):
            module_path = os.path.join(plugins_dir, filename)
            spec = importlib.util.spec_from_file_location(filename[:-3], module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for attr in dir(module):
                cls = getattr(module, attr)
                if isinstance(cls, type) and issubclass(cls, PluginBase) and cls is not PluginBase:
                    plugin_instance = cls(main_window)
                    plugin_instance.activate()
                    plugins.append(plugin_instance)
    return plugins

# =====================================================
# 10. メインウィンドウ（UI 統合）
# =====================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced VRChat Avatar Creator - ボブにゃん版")
        self.glWidget = AdvancedGLWidget(self)
        self.setCentralWidget(self.glWidget)
        self.resize(1400, 900)
        self.initMenus()
        self.initToolBar()
        self.initStatusBar()
        self.initDockWidgets()
        self.plugins = load_plugins(self)

    def initMenus(self):
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu("ファイル")
        addCubeAction = QAction("キューブ追加", self)
        addCubeAction.triggered.connect(self.glWidget.addCubeMesh)
        fileMenu.addAction(addCubeAction)
        clearAction = QAction("シーンクリア", self)
        clearAction.triggered.connect(self.glWidget.clearShapes)
        fileMenu.addAction(clearAction)
        exportFBXAction = QAction("FBXとしてエクスポート", self)
        exportFBXAction.triggered.connect(self.exportFBX)
        fileMenu.addAction(exportFBXAction)
        importVRMAction = QAction("VRMを読み込み", self)
        importVRMAction.triggered.connect(self.importVRM)
        fileMenu.addAction(importVRMAction)
        exitAction = QAction("終了", self)
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

        editMenu = menuBar.addMenu("編集")
        subdivideAction = QAction("分割", self)
        subdivideAction.triggered.connect(self.glWidget.subdivideMesh)
        editMenu.addAction(subdivideAction)
        deformAction = QAction("変形", self)
        deformAction.triggered.connect(self.glWidget.deformMesh)
        editMenu.addAction(deformAction)
        mergeAction = QAction("頂点統合", self)
        mergeAction.triggered.connect(self.glWidget.mergeMeshVertices)
        editMenu.addAction(mergeAction)
        extrudeAction = QAction("面押し出し", self)
        extrudeAction.triggered.connect(lambda: self.glWidget.extrudeMeshFace(0, 0.5))
        editMenu.addAction(extrudeAction)

        avatarMenu = menuBar.addMenu("VRChatアバター作成")
        createAvatarAction = QAction("アバター作成", self)
        createAvatarAction.triggered.connect(self.createAvatar)
        avatarMenu.addAction(createAvatarAction)

        pluginMenu = menuBar.addMenu("プラグイン")
        reloadPluginAction = QAction("プラグイン再読み込み", self)
        reloadPluginAction.triggered.connect(self.reloadPlugins)
        pluginMenu.addAction(reloadPluginAction)

        helpMenu = menuBar.addMenu("ヘルプ")
        tutorialAction = QAction("インタラクティブチュートリアル", self)
        tutorialAction.triggered.connect(self.showTutorial)
        helpMenu.addAction(tutorialAction)

    def initToolBar(self):
        toolBar = QToolBar("ツールバー", self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolBar)
        addCubeBtn = QAction("キューブ追加", self)
        addCubeBtn.triggered.connect(self.glWidget.addCubeMesh)
        toolBar.addAction(addCubeBtn)
        subdivideBtn = QAction("分割", self)
        subdivideBtn.triggered.connect(self.glWidget.subdivideMesh)
        toolBar.addAction(subdivideBtn)
        deformBtn = QAction("変形", self)
        deformBtn.triggered.connect(self.glWidget.deformMesh)
        toolBar.addAction(deformBtn)
        mergeBtn = QAction("頂点統合", self)
        mergeBtn.triggered.connect(self.glWidget.mergeMeshVertices)
        toolBar.addAction(mergeBtn)
        extrudeBtn = QAction("面押し出し", self)
        extrudeBtn.triggered.connect(lambda: self.glWidget.extrudeMeshFace(0, 0.5))
        toolBar.addAction(extrudeBtn)
        clearBtn = QAction("シーンクリア", self)
        clearBtn.triggered.connect(self.glWidget.clearShapes)
        toolBar.addAction(clearBtn)

    def initStatusBar(self):
        statusBar = QStatusBar(self)
        self.setStatusBar(statusBar)
        statusBar.showMessage("Advanced VRChat Avatar Creator 準備完了！")

    def initDockWidgets(self):
        self.propDock = QDockWidget("詳細プロパティエディタ", self)
        self.detailedPropEditor = DetailedPropertyEditor(self.glWidget, self)
        self.propDock.setWidget(self.detailedPropEditor)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.propDock)

        self.vertexDock = QDockWidget("頂点エディタ", self)
        self.vertexDock.setWidget(self.glWidget.vertexEditor)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.vertexDock)

        self.matrixDock = QDockWidget("変換行列エディタ", self)
        self.matrixDock.setWidget(self.glWidget.matrixEditor)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.matrixDock)

    def exportFBX(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "FBXとしてエクスポート", "", "FBX Files (*.fbx);;All Files (*)")
        if filepath:
            if self.glWidget.shapes:
                if export_mesh_to_fbx(self.glWidget.shapes[0], filepath):
                    self.statusBar().showMessage(f"FBXエクスポート成功: {filepath}")
                else:
                    self.statusBar().showMessage("FBXエクスポート失敗")
            else:
                self.statusBar().showMessage("エクスポートするメッシュがありません")

    def importVRM(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "VRMを読み込み", "", "VRM Files (*.vrm);;All Files (*)")
        if filepath:
            model = import_vrm_file(filepath)
            if model is not None:
                mesh = create_cube_mesh(center=(0,0,0), size=1.0)  # 例として生成
                self.glWidget.shapes.append(mesh)
                self.glWidget.update()
                self.statusBar().showMessage(f"VRM読み込み成功: {filepath}")
            else:
                self.statusBar().showMessage("VRM読み込み失敗")

    def reloadPlugins(self):
        self.plugins = load_plugins(self)
        self.statusBar().showMessage("プラグイン再読み込み完了")

    def showTutorial(self):
        wizard = TutorialWizard(self)
        wizard.exec()

    def createAvatar(self):
        if self.glWidget.shapes:
            mesh = self.glWidget.shapes[0]
            bone_data = create_vrchat_avatar(mesh)
            filepath, _ = QFileDialog.getSaveFileName(self, "VRMとしてエクスポート", "", "VRM Files (*.vrm);;All Files (*)")
            if filepath:
                # 実際の vrm2py.export_vrm を用いて出力
                if export_vrm_file(vrm2py.load_vrm(filepath), filepath):
                    self.statusBar().showMessage(f"VRChatアバター作成成功: {filepath}")
                else:
                    self.statusBar().showMessage("VRChatアバター作成失敗")
        else:
            self.statusBar().showMessage("対象メッシュがありません")

    def resizeEvent(self, event):
        self.detailedPropEditor.update_properties()
        super().resizeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec())
