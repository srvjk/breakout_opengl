from enum import Enum
import OpenGL.GL as gl
import glfw
import sys
import glm
#from stb import image as im
from PIL import Image
from numpy import asarray
import ctypes


class GameState(Enum):
    GAME_ACTIVE = 1
    GAME_MENU = 2
    GAME_WIN = 3


class Game:
    def __init__(self, width, height):
        self.game_state = GameState.GAME_ACTIVE
        self.resource_manager = ResourceManager()
        self.renderer = None
        self.width = width
        self.height = height

    def init(self):
        self.resource_manager.load_shader("sprite", "sprite.vs", "sprite.frag")
        projection = glm.ortho(0.0, self.width, self.height, 0.0)
        shader = self.resource_manager.get_shader("sprite")
        shader.use()
        shader.set_integer("image", 0)
        shader.set_matrix4("projection", projection)
        self.renderer = SpriteRenderer(shader)
        self.resource_manager.load_texture("face.png", True, "face")

    def process_input(self):
        pass

    def render(self):
        self.renderer.draw_sprite(self.resource_manager.get_texture("face"), glm.vec2(200.0, 200.0),
                                  glm.vec2(300.0, 400.0), 45.0, glm.vec3(0.0, 1.0, 0.0))


class Shader:
    def __init__(self):
        self.shader_program = None

    def compile(self, vertex_source, fragment_source, geometry_source=""):
        vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(vertex_shader, vertex_source)
        gl.glCompileShader(vertex_shader)
        compile_status = gl.glGetShaderiv(vertex_shader, gl.GL_COMPILE_STATUS)
        if not compile_status:
            print("Vertex shader compilation failed")

        fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(fragment_shader, fragment_source)
        gl.glCompileShader(fragment_shader)
        compile_status = gl.glGetShaderiv(fragment_shader, gl.GL_COMPILE_STATUS)
        if not compile_status:
            print("Fragment shader compilation failed")

        geometry_shader = None
        if geometry_source:
            geometry_shader = gl.glCreateShader(gl.GL_GEOMETRY_SHADER)
            gl.glShaderSource(geometry_shader, geometry_source)
            gl.glCompileShader(geometry_shader)
            compile_status = gl.glGetShaderiv(vertex_shader, gl.GL_COMPILE_STATUS)
            if not compile_status:
                print("Geometry shader compilation failed")

        self.shader_program = gl.glCreateProgram()
        gl.glAttachShader(self.shader_program, vertex_shader)
        gl.glAttachShader(self.shader_program, fragment_shader)
        if geometry_shader:
            gl.glAttachShader(self.shader_program, geometry_shader)
        gl.glLinkProgram(self.shader_program)
        link_status = gl.glGetProgramiv(self.shader_program, gl.GL_LINK_STATUS)
        if not link_status:
            print("Shader linking failed")

        gl.glDeleteShader(vertex_shader)
        gl.glDeleteShader(fragment_shader)
        if geometry_shader:
            gl.glDeleteShader(geometry_shader)

    def use(self):
        gl.glUseProgram(self.shader_program)

    def set_float(self, name, value, use_shader = False):
        if use_shader:
            self.use()
        gl.glUniform1f(gl.glGetUniformLocation(self.shader_program, name), value)

    def set_integer(self, name, value, use_shader = False):
        if use_shader:
            self.use()
        gl.glUniform1i(gl.glGetUniformLocation(self.shader_program, name), value)

    def set_vector2f(self, name, x, y, use_shader = False):
        if use_shader:
            self.use()
        gl.glUniform2f(gl.glGetUniformLocation(self.shader_program, name), x, y)

    def set_vector3f(self, name, x, y, z, use_shader = False):
        if use_shader:
            self.use()
        gl.glUniform3f(gl.glGetUniformLocation(self.shader_program, name), x, y, z)

    def set_vector3f(self, name, vector, use_shader = False):
        if use_shader:
            self.use()
        gl.glUniform3f(gl.glGetUniformLocation(self.shader_program, name), vector.x, vector.y, vector.z)

    def set_vector4f(self, name, x, y, z, w, use_shader = False):
        if use_shader:
            self.use()
        gl.glUniform4f(gl.glGetUniformLocation(self.shader_program, name), x, y, z, w)

    def set_vector4f(self, name, vector, use_shader = False):
        if use_shader:
            self.use()
        gl.glUniform4f(gl.glGetUniformLocation(self.shader_program, name), vector.x, vector.y, vector.z, vector.w)

    def set_matrix4(self, name, matrix, use_shader = False):
        if use_shader:
            self.use()
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.shader_program, name), 1, False, glm.value_ptr(matrix))


class Texture:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.id = gl.glGenTextures(1)
        self.internal_format = gl.GL_RGB
        self.image_format = gl.GL_RGB
        self.wrap_s = gl.GL_REPEAT
        self.wrap_t = gl.GL_REPEAT
        self.filter_min = gl.GL_LINEAR
        self.filter_max = gl.GL_LINEAR

    def generate(self, width, height, data):
        self.width = width
        self.height = height
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, self.internal_format, width, height, 0, self.image_format,
                        gl.GL_UNSIGNED_BYTE, data)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, self.wrap_s)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, self.wrap_t)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, self.filter_min)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, self.filter_max)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def bind(self):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.id)


class ResourceManager:
    def __init__(self):
        self.shaders = dict()
        self.textures = dict()

    def load_shader(self, shader_name, v_shader_file, f_shader_file, g_shader_file = None):
        v_shader_code = ""
        f_shader_code = ""
        g_shader_code = ""

        with open(v_shader_file) as f:
            v_shader_code = f.read()
        with open(f_shader_file) as f:
            f_shader_code = f.read()
        if g_shader_file:
            with open(g_shader_file) as f:
                g_shader_code = f.read()

        shader = Shader()
        shader.compile(v_shader_code, f_shader_code, g_shader_code)

        self.shaders[shader_name] = shader
        return self.shaders[shader_name]

    def load_texture(self, texture_file, alpha, texture_name):
        texture = Texture()
        if alpha:
            texture.internal_format = gl.GL_RGBA
            texture.image_format = gl.GL_RGBA

        image = Image.open(texture_file)
        data = asarray(image)
        texture.generate(data.shape[0], data.shape[1], data)

        self.textures[texture_name] = texture

        return texture

    def get_shader(self, shader_name):
        return self.shaders[shader_name]

    def get_texture(self, texture_name):
        return self.textures[texture_name]

    def clear(self):
        for k, v in self.shaders.items():
            gl.glDeleteProgram(v.id)
        for k, v in self.textures.items():
            gl.glDeleteTextures(1, v.id)


class SpriteRenderer:
    def __init__(self, shader):
        self.vao = None
        self.shader = shader
        self.init_render_data()

    def init_render_data(self):
        vertices = [
            # 1st triangle
            # pos     tex
            0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            # 2nd triangle
            # pos     tex
            0.0, 1.0, 0.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 0.0, 1.0, 0.0
        ]

        self.vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, len(vertices) * 4, (gl.GLfloat * len(vertices))(*vertices),
                        gl.GL_STATIC_DRAW)
        gl.glBindVertexArray(self.vao)

        gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, 16, None) # MAY BE WRONG!
        gl.glEnableVertexAttribArray(0)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

    def draw_sprite(self, texture, position, size, rotate, color):
        model = glm.mat4(1.0)
        model = glm.translate(model, glm.vec3(position, 0.0))
        model = glm.translate(model, glm.vec3(0.5 * size[0], 0.5 * size[1], 0.0))
        model = glm.rotate(model, glm.radians(rotate), glm.vec3(0.0, 0.0, 1.0))
        model = glm.translate(model, glm.vec3(-0.5 * size[0], -0.5 * size[1], 0.0))
        model = glm.scale(model, glm.vec3(size, 1.0))

        self.shader.set_matrix4("model", model)
        self.shader.set_vector3f("spriteColor", color)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        texture.bind()

        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)


def init_glfw():
    # Initialize the GLFW library
    if not glfw.init():
        return

    # OpenGL 3 or above is required
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # OpenGL context should be forward-compatible
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a window in windowed mode and it's OpenGL context
    primary = glfw.get_primary_monitor()  # for GLFWmonitor
    window = glfw.create_window(
        1024,  # width, is required here but overwritten by "glfw.set_window_size()" above
        768,  # height, is required here but overwritten by "glfw.set_window_size()" above
        "pyimgui-examples-glfw",  # window name, is overwritten by "glfw.set_window_title()" above
        None,  # GLFWmonitor: None = windowed mode, 'primary' to choose fullscreen (resolution needs to be adjusted)
        None  # GLFWwindow
    )

    # Exception handler if window wasn't created
    if not window:
        glfw.terminate()
        return

    # Makes window current on the calling thread
    glfw.make_context_current(window)

    # Passing window to main()
    return window


def main():
    window = init_glfw()

    win_width = 1024
    win_height = 768
    gl.glViewport(0, 0, win_width, win_height)

    attribs = gl.glGetIntegerv(gl.GL_MAX_VERTEX_ATTRIBS)
    print("Maximum number of vertex attributes supported: {}".format(attribs))

    breakout = Game(win_width, win_height)
    breakout.init()

    while not glfw.window_should_close(window):
        glfw.poll_events()
        glfw.set_window_title(window, "Breakout")
        glfw.set_window_size(window, win_width, win_height)

        breakout.process_input()

        gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        breakout.render()

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
