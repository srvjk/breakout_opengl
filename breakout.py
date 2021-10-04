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


PLAYER_SIZE = glm.vec2(128.0, 32.0)
PLAYER_VELOCITY = 500.0


class Game:
    def __init__(self, width, height):
        self.game_state = GameState.GAME_ACTIVE
        self.renderer = None
        self.width = width
        self.height = height
        self.levels = list()
        self.level = 0
        self.keys = [False for x in range(1024)]

    def init(self):
        resource_manager.load_shader("sprite", "sprite.vs", "sprite.frag")
        projection = glm.ortho(0.0, self.width, self.height, 0.0)
        shader = resource_manager.get_shader("sprite")
        shader.use()
        shader.set_integer("image", 0)
        shader.set_matrix4("projection", projection)
        self.renderer = SpriteRenderer(shader)
        resource_manager.load_texture("textures/background.jpg", False, "background")
        resource_manager.load_texture("textures/face.png", True, "face")
        resource_manager.load_texture("textures/block.png", False, "block")
        resource_manager.load_texture("textures/block_solid.png", False, "block_solid")
        resource_manager.load_texture("textures/paddle.png", True, "paddle")

        level_one = GameLevel()
        level_one.load("levels/one.lvl", self.width, self.height / 2)
        self.levels.append(level_one)
        level_two = GameLevel()
        level_two.load("levels/two.lvl", self.width, self.height / 2)
        self.levels.append(level_two)
        level_three = GameLevel()
        level_three.load("levels/three.lvl", self.width, self.height / 2)
        self.levels.append(level_three)
        level_four = GameLevel()
        level_four.load("levels/four.lvl", self.width, self.height / 2)
        self.levels.append(level_four)
        self.level = 0

        self.player = GameObject()
        self.player.position = glm.vec2(self.width / 2.0 - PLAYER_SIZE.x / 2.0, self.height - PLAYER_SIZE.y)
        self.player.size = PLAYER_SIZE
        self.player.sprite = resource_manager.get_texture("paddle")

    def process_input(self, dt):
        if self.game_state == GameState.GAME_ACTIVE:
            distance = PLAYER_VELOCITY * dt
            if self.keys[glfw.KEY_A]:
                if self.player.position.x >= 0.0:
                    self.player.position.x -= distance
            if self.keys[glfw.KEY_D]:
                if self.player.position.x <= self.width - self.player.size.x:
                    self.player.position.x += distance

    def render(self):
        if self.game_state == GameState.GAME_ACTIVE:
            self.renderer.draw_sprite(resource_manager.get_texture("background"), glm.vec2(0.0, 0.0),
                                      glm.vec2(self.width, self.height), 0.0, glm.vec3(1.0))
            self.levels[self.level].draw(self.renderer)

            self.player.draw(self.renderer)


class GameObject:
    def __init__(self):
        self.position = glm.vec2(0.0, 0.0)
        self.size = glm.vec2(1.0, 1.0)
        self.velocity = glm.vec2(0.0, 0.0)
        self.color = glm.vec3(1.0, 1.0, 1.0)
        self.rotation = 0.0
        self.is_solid = False
        self.is_destroyed = False
        self.sprite = None

    def draw(self, renderer):
        renderer.draw_sprite(self.sprite, self.position, self.size, self.rotation, self.color)


class GameLevel:
    def __init__(self):
        self.bricks = list()

    def load(self, file_path, level_width, level_height):
        self.bricks.clear()
        tile_data = list()
        with open(file_path) as f:
            for line in f:
                row = list()
                words = line.split()
                for word in words:
                    try:
                        code = int(word)
                    except ValueError:
                        return
                    row.append(code)
                tile_data.append(row)

        if tile_data:
            self._init(tile_data, level_width, level_height)

    def _init(self, tile_data, level_width, level_height):
        height = len(tile_data)
        width = len(tile_data[0])
        unit_width = level_width / float(width)
        unit_height = level_height / float(height)

        for y in range(height):
            for x in range(width):
                tile_value = tile_data[y][x]
                if tile_value == 1: # "solid" brick
                    pos = glm.vec2(unit_width * x, unit_height * y)
                    size = glm.vec2(unit_width, unit_height)
                    obj = GameObject()
                    obj.position = pos
                    obj.size = size
                    obj.sprite = resource_manager.get_texture("block_solid")
                    obj.color = glm.vec3(0.8, 0.8, 0.7)
                    obj.is_solid = True
                    self.bricks.append(obj)
                elif tile_value > 1:
                    pos = glm.vec2(unit_width * x, unit_height * y)
                    size = glm.vec2(unit_width, unit_height)
                    obj = GameObject()
                    obj.position = pos
                    obj.size = size
                    obj.sprite = resource_manager.get_texture("block")
                    color = glm.vec3(1.0)
                    if tile_value == 2:
                        color = glm.vec3(0.2, 0.6, 1.0)
                    elif tile_value == 3:
                        color = glm.vec3(0.0, 0.7, 0.0)
                    elif tile_value == 4:
                        color = glm.vec3(0.8, 0.8, 0.4)
                    elif tile_value == 5:
                        color = glm.vec3(1.0, 0.5, 0.0)
                    obj.color = color
                    self.bricks.append(obj)

    def draw(self, renderer):
        for brick in self.bricks:
            brick.draw(renderer)

    def is_completed(self):
        for brick in self.bricks:
            if (not brick.is_solid) and (not brick.is_destroyed):
                return False
        return True


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

    def set_vector3f(self, name, vector, use_shader = False):
        if use_shader:
            self.use()
        gl.glUniform3f(gl.glGetUniformLocation(self.shader_program, name), vector.x, vector.y, vector.z)

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
        texture.generate(data.shape[1], data.shape[0], data)  # Attention: in np.array height is the first param!

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

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

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

    glfw.window_hint(glfw.RESIZABLE, False)

    # Create a window in windowed mode and it's OpenGL context
    primary = glfw.get_primary_monitor()  # for GLFWmonitor
    window = glfw.create_window(
        SCREEN_WIDTH,  # width, is required here but overwritten by "glfw.set_window_size()" above
        SCREEN_HEIGHT,  # height, is required here but overwritten by "glfw.set_window_size()" above
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

    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    glfw.set_key_callback(window, key_callback)

    # Passing window to main()
    return window

breakout = Game(SCREEN_WIDTH, SCREEN_HEIGHT)
resource_manager = ResourceManager()

def framebuffer_size_callback(window, width, height):
    gl.glViewport(0, 0, win_width, win_height)

def key_callback(window, key, scancode, action, mods):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.window_should_close(window)
    if key >= 0 and key < 1024:
        if action == glfw.PRESS:
            breakout.keys[key] = True
        elif action == glfw.RELEASE:
            breakout.keys[key] = False

def main():
    window = init_glfw()

    win_width = SCREEN_WIDTH
    win_height = SCREEN_HEIGHT
    gl.glViewport(0, 0, win_width, win_height)

    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    attribs = gl.glGetIntegerv(gl.GL_MAX_VERTEX_ATTRIBS)
    print("Maximum number of vertex attributes supported: {}".format(attribs))

    breakout.init()

    delta_time = 0.0
    last_time = 0.0

    while not glfw.window_should_close(window):
        cur_time = glfw.get_time()
        delta_time = cur_time - last_time
        last_time = cur_time

        glfw.poll_events()
        glfw.set_window_title(window, "Breakout")
        glfw.set_window_size(window, win_width, win_height)

        breakout.process_input(delta_time)

        gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        breakout.render()

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
