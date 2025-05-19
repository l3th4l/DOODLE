import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import compileShader, compileProgram
import numpy as np
import math

# Globals for shader program and uniform locations
shader = None
u_lightDir = None
u_lightColor = None
u_ambientColor = None
u_materialDiffuse = None
u_materialSpecular = None
u_shininess = None

# Utility: normalize a vector
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

# Builds a shadow projection matrix onto the ground plane (y=0) from a light direction
def shadow_matrix(light_dir):
    # Plane: y=0 => equation 0*x+1*y+0*z+0=0
    a, b, c, d = 0.0, 1.0, 0.0, 0.0
    lx, ly, lz = light_dir
    dot = a*lx + b*ly + c*lz + d
    M = np.array([
        [dot - a*lx,   -a*ly,     -a*lz,     -a*d],
        [  -b*lx,   dot - b*ly,   -b*lz,     -b*d],
        [  -c*lx,      -c*ly,  dot - c*lz,   -c*d],
        [  -d*lx,      -d*ly,     -d*lz,  dot - d ]
    ], dtype=np.float32)
    return M.T

# GLSL shaders (Phong lighting)
VERTEX_SHADER = """
#version 120
varying vec3 fragPos;
varying vec3 fragNormal;
void main() {
    fragPos = vec3(gl_ModelViewMatrix * gl_Vertex);
    fragNormal = normalize(gl_NormalMatrix * gl_Normal);
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;
}
"""

FRAGMENT_SHADER = """
#version 120
varying vec3 fragPos;
varying vec3 fragNormal;
uniform vec3 lightDir;
uniform vec3 lightColor;
uniform vec3 ambientColor;
uniform vec3 materialDiffuse;
uniform vec3 materialSpecular;
uniform float shininess;
void main() {
    vec3 N = normalize(fragNormal);
    vec3 L = normalize(-lightDir);
    vec3 ambient = ambientColor * materialDiffuse;
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * lightColor * materialDiffuse;
    vec3 V = normalize(-fragPos);
    vec3 R = reflect(lightDir, N);
    float spec = pow(max(dot(V, R), 0.0), shininess);
    vec3 specular = spec * lightColor * materialSpecular;
    gl_FragColor = vec4(ambient + diffuse + specular, 1.0);
}
"""

# Draw central tower
def draw_tower():
    glUniform3f(u_materialDiffuse, 0.6, 0.6, 0.6)
    glUniform3f(u_materialSpecular, 0.3, 0.3, 0.3)
    glUniform1f(u_shininess, 50.0)
    quad = gluNewQuadric()
    glPushMatrix()
    glRotatef(-90, 1, 0, 0)
    gluCylinder(quad, 1, 1, 5, 32, 32)
    glTranslatef(0, 0, 5)
    gluDisk(quad, 0, 1, 32, 1)
    glPopMatrix()

# Draw reflective plane
def draw_plane(pos, normal, area):
    half = math.sqrt(area)/2
    up = np.array([0,1,0])
    axis = np.cross(up, normal)
    ang = math.degrees(math.acos(np.dot(up, normal)))
    glUniform3f(u_materialDiffuse, 1.0, 1.0, 1.0)
    glUniform3f(u_materialSpecular, 1.0, 1.0, 1.0)
    glUniform1f(u_shininess, 128.0)
    glPushMatrix()
    glTranslatef(*pos)
    if np.linalg.norm(axis)>1e-6:
        ax = normalize(axis)
        glRotatef(ang, ax[0], ax[1], ax[2])
    glBegin(GL_QUADS)
    glNormal3f(*normal)
    glVertex3f(-half,0,-half)
    glVertex3f( half,0,-half)
    glVertex3f( half,0, half)
    glVertex3f(-half,0, half)
    glEnd()
    glPopMatrix()

# Draw ground
def draw_ground(size=50):
    glUniform3f(u_materialDiffuse, 0.3, 0.5, 0.3)
    glUniform3f(u_materialSpecular, 0.0, 0.0, 0.0)
    glUniform1f(u_shininess, 1.0)
    glBegin(GL_QUADS)
    glNormal3f(0,1,0)
    glVertex3f(-size,0,-size)
    glVertex3f( size,0,-size)
    glVertex3f( size,0, size)
    glVertex3f(-size,0, size)
    glEnd()

# Setup shaders
def setup_shaders(sun):
    global shader, u_lightDir, u_lightColor, u_ambientColor
    global u_materialDiffuse, u_materialSpecular, u_shininess
    shader = compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )
    glUseProgram(shader)
    u_lightDir = glGetUniformLocation(shader, "lightDir")
    u_lightColor = glGetUniformLocation(shader, "lightColor")
    u_ambientColor = glGetUniformLocation(shader, "ambientColor")
    u_materialDiffuse = glGetUniformLocation(shader, "materialDiffuse")
    u_materialSpecular = glGetUniformLocation(shader, "materialSpecular")
    u_shininess = glGetUniformLocation(shader, "shininess")
    glUniform3f(u_lightDir, *sun)
    glUniform3f(u_lightColor, 1.0, 1.0, 0.95)
    glUniform3f(u_ambientColor, 0.2, 0.2, 0.2)

# Main function with stencil-based shadows
def main():
    pygame.init()
    display = (800, 600)

    # ── Request a stencil buffer and a decent depth buffer ─────────────────
    pygame.display.gl_set_attribute(pygame.GL_STENCIL_SIZE, 8)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)

    # ── Now open an OpenGL window with double buffering ────────────────────
    pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)

    # your existing setup: viewport, projection, shaders, etc.
    glViewport(0, 0, display[0], display[1])
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_STENCIL_TEST)
    setup_shaders()
    setup_scene()

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # ── 1) Clear depth, color, AND stencil ──────────────────────────────
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

        # ── 2) Stencil pass: mark ground pixels ────────────────────────────
        glStencilFunc(GL_ALWAYS, 1, 0xFF)            # all fragments pass
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE)    # replace stencil with 1
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)  # no color writes
        draw_ground()                                # writes 1 into stencil

        # ── 3) Shadow pass: only where stencil==1 ─────────────────────────
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
        glStencilFunc(GL_EQUAL, 1, 0xFF)             # pass only if stencil==1
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)       # keep stencil buffer
        draw_shadows()                               # projected geometry

        # ── 4) Final pass: draw the real scene ────────────────────────────
        glStencilFunc(GL_ALWAYS, 0, 0xFF)            # no stencil clipping
        draw_ground()
        draw_tower_and_reflectors()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__": main()
