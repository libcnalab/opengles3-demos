// es32-debug-output
//
// Debug output of OpenGL ES 3.2 (ES 3.2 chapter 18, section 20.2 GetPointerv). The GL reports what it is doing, and
// applications can annotate their own command stream, through a stream of *debug messages*: each one has a source,
// a type, an id, a severity and a string. The demo exercises every piece of that machinery in a debug context
// (EGL_CONTEXT_OPENGL_DEBUG, requested through demo::Info::debug_context):
//   - delivery: glDebugMessageCallback with a user pointer (read back with glGetPointerv), DEBUG_OUTPUT and
//     DEBUG_OUTPUT_SYNCHRONOUS, application messages through glDebugMessageInsert received with their exact
//     source / type / id / severity / length / text;
//   - filtering: glDebugMessageControl by source, type, severity and by lists of ids, the initial state (everything is
//     enabled except severity LOW), and the defined errors;
//   - the message log (no callback): GetDebugMessageLog, DEBUG_LOGGED_MESSAGES, DEBUG_NEXT_LOGGED_MESSAGE_LENGTH and
//     what happens when MAX_DEBUG_LOGGED_MESSAGES is exceeded;
//   - debug groups: glPushDebugGroup / glPopDebugGroup generate messages, carry their own filter state, and have a
//     depth limit (STACK_OVERFLOW / STACK_UNDERFLOW);
//   - labels: glObjectLabel / glGetObjectLabel for all 11 object namespaces of Table 18.4, glObjectPtrLabel /
//     glGetObjectPtrLabel for a sync object, MAX_LABEL_LENGTH, truncation by bufSize and the INVALID_VALUE cases.
//
// Picture (256x256, rows are numbered from the bottom):
//   y 144..255  a tiny scene (flat triangle, 2x2 checker quad) drawn with labelled objects inside a debug group
//   y 128..143  five squares: the labels of the five scene objects read back after the frame (green = as set)
//   y   0..127  32 cells (8 x 4, 32 px each), one per validated area in the order of validate(); green = every check
//               of the area passed, red = at least one failed, dark = unused

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace mgl = metagl;

namespace
{
    using EC = mgl::ErrorCode;
    using Src = mgl::DebugSource;
    using Typ = mgl::DebugType;
    using Sev = mgl::DebugSeverity;
    using Ns = mgl::DebugObjectLabel;
    using Rgba8 = demo::Rgba8;

    // ---------------------------------------------------------------------------------------------------------
    // The debug message stream as the callback sees it.
    // ---------------------------------------------------------------------------------------------------------

    /// One message exactly as delivered to the callback (ES 3.2 section 18.2).
    struct Message
    {
        Src source{};
        Typ type{};
        GLuint id = 0;
        Sev severity{};
        GLsizei length = 0;           // the `length` argument of the callback (excluding the null terminator)
        std::string text;             // the `message` string
        bool terminated = false;      // strlen(message) == length
        const void* user = nullptr;   // the `userParam` argument
        int via = 0;                  // which callback function delivered it (1 = A, 2 = B)
        std::thread::id thread;       // the thread that ran the callback
    };

    /// Where the callbacks put what they receive. Locked, because with DEBUG_OUTPUT_SYNCHRONOUS disabled the GL is
    /// allowed to call back from other threads (section 18.8); the demo runs with synchronous output enabled.
    class Sink
    {
    public:
        void push(Message m) const
        {
            std::lock_guard lock(mutex_);
            messages_.push_back(std::move(m));
        }
        [[nodiscard]] std::vector<Message> take() const
        {
            std::lock_guard lock(mutex_);
            return std::exchange(messages_, std::vector<Message>{});
        }
        [[nodiscard]] std::vector<Message> snapshot() const
        {
            std::lock_guard lock(mutex_);
            return messages_;
        }
        void clear() const
        {
            std::lock_guard lock(mutex_);
            messages_.clear();
        }
        [[nodiscard]] std::size_t size() const
        {
            std::lock_guard lock(mutex_);
            return messages_.size();
        }

    private:
        mutable std::mutex mutex_;
        mutable std::vector<Message> messages_;
    };

    // The callback type GLDEBUGPROC is declared with the platform calling convention GL_APIENTRY (section 18.2: "callback
    // must be declared with the same platform-dependent calling convention ... Anything else will result in undefined
    // behavior"). The macro comes from the GL headers that meta-gl includes. The arguments arrive as raw GLenum values;
    // they are always members of Tables 18.1-18.3, so converting them to the typed enums is safe.
    // The callback makes no GL call (section 18.2: undefined behavior).
    void deliver(int via, const void* user, GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
                 const GLchar* message)
    {
        if (user == nullptr) return;
        Message m;
        m.source = static_cast<Src>(source);
        m.type = static_cast<Typ>(type);
        m.id = id;
        m.severity = static_cast<Sev>(severity);
        m.length = length;
        // The text is read by `length`, which is always safe. Whether the string is also null-terminated at `length`
        // (section 18.1: "null-terminated string representation") is recorded separately. The demo's own strings are
        // always terminated, so strlen() stays inside the caller's buffer.
        m.text = message != nullptr && length > 0 ? std::string(message, static_cast<std::size_t>(length)) : std::string{};
        m.terminated = message != nullptr && std::strlen(message) == static_cast<std::size_t>(length);
        m.user = user;
        m.via = via;
        m.thread = std::this_thread::get_id();
        static_cast<const Sink*>(user)->push(std::move(m));
    }

    void GL_APIENTRY debug_callback_a(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
                                      const GLchar* message, const void* userParam)
    {
        deliver(1, userParam, source, type, id, severity, length, message);
    }

    void GL_APIENTRY debug_callback_b(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
                                      const GLchar* message, const void* userParam)
    {
        deliver(2, userParam, source, type, id, severity, length, message);
    }

    std::string describe(const Message& m)
    {
        return std::format("[{} {} id={} {} len={} '{}'{}]", mgl::to_string(m.source), mgl::to_string(m.type), m.id,
                           mgl::to_string(m.severity), m.length, m.text, m.terminated ? "" : " NOT-TERMINATED");
    }

    std::string describe(const std::vector<Message>& v)
    {
        std::string s = std::format("{} message(s)", v.size());
        for (const Message& m : v) s += " " + describe(m);
        return s;
    }

    std::string describe_ids(const std::vector<GLuint>& ids)
    {
        std::string s = "{";
        for (std::size_t i = 0; i < ids.size(); ++i) s += std::format("{}{}", i == 0 ? "" : ",", ids[i]);
        return s + "}";
    }

    /// Does the message have exactly these fields? The text, its `length` and (normally) the terminator must agree.
    bool matches(const Message& m, Src source, Typ type, GLuint id, Sev severity, std::string_view text,
                 bool need_terminator = true)
    {
        return m.source == source && m.type == type && m.id == id && m.severity == severity && m.text == text &&
               m.length == static_cast<GLsizei>(text.size()) && (m.terminated || !need_terminator);
    }

    /// Exactly one message, with exactly these fields.
    bool expect_one(demo::Check& check, const std::vector<Message>& got, std::string_view what, Src source, Typ type,
                    GLuint id, Sev severity, std::string_view text, bool need_terminator = true)
    {
        const bool ok = got.size() == 1 && matches(got[0], source, type, id, severity, text, need_terminator);
        return check.expect(ok, what, ok ? std::string{} : describe(got));
    }

    /// Mesa hands the callback the caller's own pointer when an explicit `length` is shorter than the string, so the
    /// message is not terminated at `length` (docs/KNOWN_LIMITATIONS.md).
    void expect_terminated_at_length(demo::Check& check, const std::vector<Message>& got, std::string_view what)
    {
        check.expect_or_known_deviation(got.size() == 1 && got[0].terminated, what, got.empty() ? std::string{} : describe(got[0]), "Mesa",
                                        "the callback receives the caller's pointer, which is only terminated after the full string, not at `length`");
    }

    std::vector<GLuint> ids_of(const std::vector<Message>& v)
    {
        std::vector<GLuint> ids;
        for (const Message& m : v) ids.push_back(m.id);
        return ids;
    }

    std::vector<GLuint> ids_of(const std::vector<Message>& v, Src source, Typ type)
    {
        std::vector<GLuint> ids;
        for (const Message& m : v)
            if (m.source == source && m.type == type) ids.push_back(m.id);
        return ids;
    }

    std::vector<Message> only_type(const std::vector<Message>& v, Typ type)
    {
        std::vector<Message> out;
        for (const Message& m : v)
            if (m.type == type) out.push_back(m);
        return out;
    }

    // ---------------------------------------------------------------------------------------------------------
    // Thin GL helpers (every call is still an explicit metagl::gl* call).
    // ---------------------------------------------------------------------------------------------------------

    GLint geti(mgl::GetParameter pname) { return demo::get_integer(pname); }

    /// glDebugMessageInsert with a null-terminated string (length < 0).
    void insert_z(Src source, Typ type, GLuint id, Sev severity, const char* text)
    {
        mgl::glDebugMessageInsert(source, type, id, severity, -1, text);
    }

    /// glDebugMessageInsert with an explicit length.
    void insert_n(Src source, Typ type, GLuint id, Sev severity, std::string_view text)
    {
        mgl::glDebugMessageInsert(source, type, id, severity, static_cast<GLsizei>(text.size()), text.data());
    }

    void control_all(bool enabled)
    {
        mgl::glDebugMessageControl(Src::DontCare, Typ::DontCare, Sev::DontCare, 0, nullptr, enabled ? GL_TRUE : GL_FALSE);
    }

    void callback_pointers(GLDEBUGPROC& function, void*& user)
    {
        void* p = nullptr;
        mgl::glGetPointerv(mgl::GetPointerParameter::DebugCallbackFunction, &p);
        function = nullptr;
        // A function pointer is stored in a void*: copy the representation (both have the size of a code address).
        static_assert(sizeof(function) == sizeof(p));
        std::memcpy(&function, &p, sizeof function);
        user = nullptr;
        mgl::glGetPointerv(mgl::GetPointerParameter::DebugCallbackUserParam, &user);
    }

    /// Empties the message log (null arrays: the messages are removed, their attributes discarded; section 18.9).
    void drain_log()
    {
        const GLuint max = static_cast<GLuint>(geti(mgl::GetParameter::MaxDebugLoggedMessages));
        (void)mgl::glGetDebugMessageLog(max, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    /// The result of one glGetDebugMessageLog call with all arrays requested.
    struct Fetched
    {
        GLuint count = 0;
        std::vector<Src> sources;
        std::vector<Typ> types;
        std::vector<GLuint> ids;
        std::vector<Sev> severities;
        std::vector<GLsizei> lengths;
        std::vector<char> text;
    };

    Fetched fetch_log(GLuint count, GLsizei buf_size)
    {
        Fetched f;
        f.sources.assign(count, Src::DontCare);
        f.types.assign(count, Typ::DontCare);
        f.ids.assign(count, 0);
        f.severities.assign(count, Sev::DontCare);
        f.lengths.assign(count, -1);
        f.text.assign(static_cast<std::size_t>(std::max(buf_size, 0)), '\x7f');
        f.count = mgl::glGetDebugMessageLog(count, buf_size, f.sources.data(), f.types.data(), f.ids.data(),
                                            f.severities.data(), f.lengths.data(), f.text.data());
        return f;
    }

    // ---------------------------------------------------------------------------------------------------------
    // Labels
    // ---------------------------------------------------------------------------------------------------------

    struct LabelRead
    {
        std::string text;              // up to the first null terminator
        GLsizei length = -1;           // what the GL reported
        bool terminated = false;       // a null terminator exists within bufSize
        bool tail_intact = true;       // nothing was written at or beyond buf[bufSize]
    };

    using SetLabel = std::function<void(GLsizei length, const GLchar* label)>;
    using GetLabel = std::function<void(GLsizei buf_size, GLsizei* length, GLchar* label)>;

    LabelRead read_label(const GetLabel& get, GLsizei buf_size)
    {
        constexpr char kSentinel = '\x7f';
        std::vector<char> buf(static_cast<std::size_t>(buf_size) + 8, kSentinel);
        LabelRead r;
        get(buf_size, &r.length, buf.data());
        const auto end = buf.begin() + buf_size;
        const auto nul = std::find(buf.begin(), end, '\0');
        r.terminated = nul != end;
        r.text.assign(buf.begin(), nul);
        r.tail_intact = std::all_of(end, buf.end(), [](char c) { return c == kSentinel; });
        return r;
    }

    std::string describe(const LabelRead& r)
    {
        return std::format("text='{}' length={} terminated={} tail_intact={}", r.text, r.length, r.terminated, r.tail_intact);
    }

    bool label_is(const LabelRead& r, std::string_view text)
    {
        return r.text == text && r.length == static_cast<GLsizei>(text.size()) && r.terminated && r.tail_intact;
    }

    /// The object namespaces of Table 18.4, each with a way to make a real object and delete it.
    struct LabelSubject
    {
        const char* name;
        Ns ns;
        std::function<GLuint()> create;
        std::function<void(GLuint)> destroy;
    };

    // ---------------------------------------------------------------------------------------------------------
    // Areas and the panel
    // ---------------------------------------------------------------------------------------------------------

    struct AreaResult
    {
        std::string name;
        int checks = 0;
        int failed = 0;
    };

    // ---------------------------------------------------------------------------------------------------------
    // Scene
    // ---------------------------------------------------------------------------------------------------------

    struct Vertex
    {
        float x, y, u, v;
    };

    constexpr std::array<Vertex, 7> kVertices = {{
        // triangle (first 3)
        {-0.9f, -0.8f, 0.0f, 0.0f},
        {-0.1f, -0.8f, 1.0f, 0.0f},
        {-0.5f, +0.8f, 0.5f, 1.0f},
        // quad as a triangle strip (last 4): x in [0.1, 0.9], y in [-0.8, 0.8]
        {0.1f, -0.8f, 0.0f, 0.0f},
        {0.9f, -0.8f, 1.0f, 0.0f},
        {0.1f, +0.8f, 0.0f, 1.0f},
        {0.9f, +0.8f, 1.0f, 1.0f},
    }};

    constexpr std::array<std::uint8_t, 16> kTexels = {
        255, 0,   0,   255,  // (0,0) red      lower left
        0,   255, 0,   255,  // (1,0) green    lower right
        0,   0,   255, 255,  // (0,1) blue     upper left
        255, 255, 0,   255,  // (1,1) yellow   upper right
    };

    constexpr std::array<float, 4> kTriangleColor = {1.0f, 0.6f, 0.2f, 1.0f};  // (255, 153, 51)
    constexpr std::array<float, 4> kClear = {0.10f, 0.10f, 0.12f, 1.0f};

    constexpr int kPanelColumns = 8;
    constexpr int kPanelRows = 4;
    constexpr int kCell = 32;
    constexpr int kStripHeight = 16;

    // The ids/texts of the scene's own debug annotations.
    constexpr GLuint kSceneGroupId = 100;
    constexpr GLuint kMarkerTriangleId = 101;
    constexpr GLuint kMarkerQuadId = 102;

    std::uint8_t to8(float v) { return static_cast<std::uint8_t>(std::lround(v * 255.0f)); }

    constexpr std::array<const char*, 5> kSceneLabels = {"scene program", "scene vertex array", "scene vertices",
                                                         "checker texture", "nearest sampler"};

    class DebugOutput final : public demo::Demo
    {
    public:
        // ---------------------------------------------------------------------------------------------------------
        void init(demo::Env& env) override
        {
            // 1. Snapshot of the initial debug state, before anything touches it (Tables 21.53 and 21.54).
            initial_.flags = geti(mgl::GetParameter::ContextFlags);
            initial_.debug_output = mgl::glIsEnabled(mgl::Capability::DebugOutput);
            initial_.synchronous = mgl::glIsEnabled(mgl::Capability::DebugOutputSynchronous);
            callback_pointers(initial_.callback, initial_.user);
            initial_.depth = geti(mgl::GetParameter::DebugGroupStackDepth);
            initial_.logged = geti(mgl::GetParameter::DebugLoggedMessages);
            initial_.next_length = geti(mgl::GetParameter::DebugNextLoggedMessageLength);
            initial_.max_message_length = geti(mgl::GetParameter::MaxDebugMessageLength);
            initial_.max_logged = geti(mgl::GetParameter::MaxDebugLoggedMessages);
            initial_.max_depth = geti(mgl::GetParameter::MaxDebugGroupStackDepth);
            initial_.max_label = geti(mgl::GetParameter::MaxLabelLength);

            // 2. The initial enabled state of messages (section 18.1: "All messages are initially enabled unless their
            //    assigned severity is DEBUG_SEVERITY_LOW"), observed before any glDebugMessageControl call: first
            //    through the message log (no callback is set yet), then through a callback. Synchronous output makes
            //    the result available when the insert returns (section 18.8).
            mgl::glEnable(mgl::Capability::DebugOutputSynchronous);
            const std::array<Sev, 4> severities = {Sev::High, Sev::Medium, Sev::Low, Sev::Notification};
            for (std::size_t i = 0; i < severities.size(); ++i)
                insert_z(Src::Application, Typ::Other, 0x700 + static_cast<GLuint>(i), severities[i], "default state");
            initial_.log_count_after_inserts = geti(mgl::GetParameter::DebugLoggedMessages);
            const Fetched f = fetch_log(16, 4096);
            initial_.log_ids = std::vector<GLuint>(f.ids.begin(), f.ids.begin() + f.count);
            initial_.log_sources_application = std::all_of(f.sources.begin(), f.sources.begin() + f.count,
                                                           [](Src s) { return s == Src::Application; });

            mgl::glDebugMessageCallback(&debug_callback_a, &sink_);
            for (std::size_t i = 0; i < severities.size(); ++i)
                insert_z(Src::ThirdParty, Typ::Marker, 0x710 + static_cast<GLuint>(i), severities[i], "default state");
            initial_.callback_ids = ids_of(sink_.take());

            // 3. The scene's objects, each with a debug label.
            program_ = demo::build_program(env, "shaders/scene.vert", "shaders/scene.frag");
            mgl::glObjectLabel(Ns::Program, program_.value, -1, kSceneLabels[0]);

            mgl::glGenVertexArrays(1, &vao_);
            mgl::glBindVertexArray(vao_);
            mgl::glObjectLabel(Ns::VertexArray, vao_.value, -1, kSceneLabels[1]);

            mgl::glGenBuffers(1, &vbo_);
            mgl::glBindBuffer(mgl::BufferTarget::Array, vbo_);
            mgl::glBufferData(mgl::BufferTarget::Array, std::span<const Vertex>(kVertices), mgl::BufferUsage::StaticDraw);
            mgl::glObjectLabel(Ns::Buffer, vbo_.value, -1, kSceneLabels[2]);
            const mgl::AttribLocation a_pos{0u};
            const mgl::AttribLocation a_uv{1u};
            mgl::glEnableVertexAttribArray(a_pos);
            mgl::glVertexAttribPointer(a_pos, 2, mgl::DataType::Float, GL_FALSE, sizeof(Vertex),
                                       demo::buffer_offset(offsetof(Vertex, x)));
            mgl::glEnableVertexAttribArray(a_uv);
            mgl::glVertexAttribPointer(a_uv, 2, mgl::DataType::Float, GL_FALSE, sizeof(Vertex),
                                       demo::buffer_offset(offsetof(Vertex, u)));
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});

            mgl::glGenTextures(1, &texture_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, texture_);
            mgl::glTexStorage2D(mgl::TextureTarget::Texture2D, 1, mgl::SizedInternalFormat::Rgba8, 2, 2);
            mgl::glTexSubImage2D(mgl::TextureTarget::Texture2D, 0, 0, 0, 2, 2, mgl::PixelFormat::Rgba,
                                 mgl::PixelType::UnsignedByte, std::span<const std::uint8_t>(kTexels));
            mgl::glObjectLabel(Ns::Texture, texture_.value, -1, kSceneLabels[3]);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});

            mgl::glGenSamplers(1, &sampler_);
            mgl::glBindSampler(0, sampler_);
            mgl::glSamplerParameteri(sampler_, mgl::SamplerParameter::MinFilter, static_cast<GLint>(GL_NEAREST));
            mgl::glSamplerParameteri(sampler_, mgl::SamplerParameter::MagFilter, static_cast<GLint>(GL_NEAREST));
            mgl::glSamplerParameteri(sampler_, mgl::SamplerParameter::WrapS, static_cast<GLint>(GL_CLAMP_TO_EDGE));
            mgl::glSamplerParameteri(sampler_, mgl::SamplerParameter::WrapT, static_cast<GLint>(GL_CLAMP_TO_EDGE));
            mgl::glObjectLabel(Ns::Sampler, sampler_.value, -1, kSceneLabels[4]);
            mgl::glBindSampler(0, mgl::SamplerId{0});
        }

        // ---------------------------------------------------------------------------------------------------------
        void render(demo::Env& env, int /*frame*/) override
        {
            // Only the messages of this frame are kept (a windowed run renders forever).
            sink_.clear();

            const int w = env.width;
            const int h = env.height;
            scene_y_ = kPanelRows * kCell + kStripHeight;
            scene_h_ = std::max(h - scene_y_, 1);

            mgl::glViewport(0, 0, w, h);
            mgl::glClearColor(kClear[0], kClear[1], kClear[2], kClear[3]);
            mgl::glClear(mgl::ClearBufferBit::Color);

            // The scene is annotated with a debug group and two markers (sections 18.5 and 18.6). They are part of
            // the command stream and change no pixel.
            mgl::glPushDebugGroup(Src::Application, kSceneGroupId, -1, "Scene");
            mgl::glViewport(0, scene_y_, w, scene_h_);
            mgl::glUseProgram(program_);
            mgl::glBindVertexArray(vao_);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, texture_);
            mgl::glBindSampler(0, sampler_);

            insert_z(Src::Application, Typ::Marker, kMarkerTriangleId, Sev::Notification, "triangle");
            mgl::glUniform4f(mgl::UniformLocation{0}, kTriangleColor[0], kTriangleColor[1], kTriangleColor[2], kTriangleColor[3]);
            mgl::glUniform1i(mgl::UniformLocation{1}, 0);
            mgl::glDrawArrays(mgl::PrimitiveType::Triangles, 0, 3);

            insert_z(Src::Application, Typ::Marker, kMarkerQuadId, Sev::Notification, "quad");
            mgl::glUniform1i(mgl::UniformLocation{1}, 1);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 3, 4);

            mgl::glBindSampler(0, mgl::SamplerId{0});
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glBindVertexArray(mgl::VertexArrayId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glViewport(0, 0, w, h);
            mgl::glPopDebugGroup();
        }

        // ---------------------------------------------------------------------------------------------------------
        void validate(demo::Env& env, demo::Check& check) override
        {
            // What the scene's render() put into the stream, captured before any area changes the debug state.
            scene_stream_ = sink_.snapshot();

            run_area(check, "context and initial state", [&] { area_context(check); });
            run_area(check, "initial message state", [&] { area_defaults(check); });
            run_area(check, "callback + GetPointerv", [&] { area_pointers(check); });
            run_area(check, "DebugMessageInsert", [&] { area_insert(check); });
            run_area(check, "Insert errors and limits", [&] { area_insert_errors(check); });
            run_area(check, "DEBUG_OUTPUT / SYNC", [&] { area_output_enable(check); });
            run_area(check, "Control by source/type/sev", [&] { area_control_filters(check); });
            run_area(check, "Control by id lists", [&] { area_control_ids(check); });
            run_area(check, "Control errors", [&] { area_control_errors(check); });
            run_area(check, "message log", [&] { area_log(check); });
            run_area(check, "log overflow", [&] { area_log_overflow(check); });
            run_area(check, "debug group messages", [&] { area_group_messages(check); });
            run_area(check, "debug group filters", [&] { area_group_filters(check); });
            run_area(check, "group depth limits", [&] { area_group_limits(check); });
            run_area(check, "GL error messages", [&] { area_error_messages(check); });
            run_area(check, "scene stream + pixels", [&] { area_scene(env, check); });
            run_area(check, "scene object labels", [&] { area_scene_labels(check); });

            for (const LabelSubject& s : label_subjects())
                run_area(check, std::format("labels: {}", s.name), [&] { area_labels(check, s); });
            run_area(check, "labels: sync object", [&] { area_sync_labels(check); });

            run_area(check, "negative controls", [&] { area_negative(env, check); });

            // Leave the context in a tidy state and show the results.
            reset_state();
            draw_panel(env);
            print_summary();
        }

        // ---------------------------------------------------------------------------------------------------------
        void shutdown(demo::Env& /*env*/) override
        {
            // The sink dies with this object: stop the GL from calling back first.
            mgl::glDebugMessageCallback(nullptr, nullptr);
            mgl::glDisable(mgl::Capability::DebugOutputSynchronous);

            mgl::glDeleteProgram(program_);
            mgl::glDeleteVertexArrays(1, &vao_);
            mgl::glDeleteBuffers(1, &vbo_);
            mgl::glDeleteTextures(1, &texture_);
            mgl::glDeleteSamplers(1, &sampler_);
            if (initial_.max_logged > 0) drain_log();
        }

    private:
        // ---------------------------------------------------------------------------------------------------------
        // Infrastructure
        // ---------------------------------------------------------------------------------------------------------

        /// Runs one area with a clean debug state and counts its checks. Every area ends with a "no GL error" check.
        void run_area(demo::Check& check, const std::string& name, const std::function<void()>& body)
        {
            std::cout << std::format("\n -- {} --\n", name);
            const int total0 = check.total();
            const int failed0 = check.failed();
            reset_state();
            body();
            check.expect_no_gl_error(std::format("[{}] leaves no GL error", name));
            areas_.push_back({name, check.total() - total0, check.failed() - failed0});
        }

        /// A known baseline: output on, synchronous, callback A with the main sink, every group popped, and only
        /// application and third-party messages enabled (all types, all severities), so that messages the
        /// implementation itself generates cannot disturb the expected message sequences.
        void reset_state()
        {
            while (geti(mgl::GetParameter::DebugGroupStackDepth) > 1) mgl::glPopDebugGroup();
            mgl::glEnable(mgl::Capability::DebugOutput);
            mgl::glEnable(mgl::Capability::DebugOutputSynchronous);
            mgl::glDebugMessageCallback(&debug_callback_a, &sink_);
            control_all(false);
            mgl::glDebugMessageControl(Src::Application, Typ::DontCare, Sev::DontCare, 0, nullptr, GL_TRUE);
            mgl::glDebugMessageControl(Src::ThirdParty, Typ::DontCare, Sev::DontCare, 0, nullptr, GL_TRUE);
            drain_log();
            sink_.clear();
            sink_b_.clear();
            (void)demo::drain_gl_errors();
        }

        /// Inserts one application message and reports whether it reached the sink (and nothing else did).
        bool delivered(Src source, Typ type, GLuint id, Sev severity)
        {
            sink_.clear();
            insert_z(source, type, id, severity, "probe");
            const std::vector<Message> got = sink_.take();
            return got.size() == 1 && matches(got[0], source, type, id, severity, "probe");
        }

        // ---------------------------------------------------------------------------------------------------------
        // Areas
        // ---------------------------------------------------------------------------------------------------------

        void area_context(demo::Check& check)
        {
            const Initial& i = initial_;
            check.expect((i.flags & static_cast<GLint>(GL_CONTEXT_FLAG_DEBUG_BIT)) != 0,
                         "CONTEXT_FLAGS contains CONTEXT_FLAG_DEBUG_BIT (EGL_CONTEXT_OPENGL_DEBUG was requested; ES 3.2 §20.2)",
                         std::format("CONTEXT_FLAGS = {:#x}", i.flags));
            check.expect(i.debug_output, "DEBUG_OUTPUT is initially TRUE in a debug context (§18, Table 21.54)");
            check.expect(!i.synchronous, "DEBUG_OUTPUT_SYNCHRONOUS is initially FALSE (§18.8)");
            check.expect(i.callback == nullptr && i.user == nullptr,
                         "DEBUG_CALLBACK_FUNCTION and DEBUG_CALLBACK_USER_PARAM are initially NULL (Table 21.53)");
            check.expect_eq(i.depth, 1, "DEBUG_GROUP_STACK_DEPTH is initially 1: only the default group (Table 21.53)");
            check.expect_eq(i.logged, 0, "DEBUG_LOGGED_MESSAGES is initially 0 (Table 21.53)");
            check.expect_eq(i.next_length, 0, "DEBUG_NEXT_LOGGED_MESSAGE_LENGTH is initially 0 (Table 21.53)");
            check.expect(i.max_message_length >= 1, "MAX_DEBUG_MESSAGE_LENGTH >= 1 (Table 21.54)",
                         std::format("{}", i.max_message_length));
            check.expect(i.max_logged >= 1, "MAX_DEBUG_LOGGED_MESSAGES >= 1 (Table 21.54)", std::format("{}", i.max_logged));
            check.expect(i.max_depth >= 64, "MAX_DEBUG_GROUP_STACK_DEPTH >= 64 (Table 21.54)", std::format("{}", i.max_depth));
            check.expect(i.max_label >= 256, "MAX_LABEL_LENGTH >= 256 (Table 21.54)", std::format("{}", i.max_label));
            std::cout << std::format("   note: MAX_DEBUG_MESSAGE_LENGTH={} MAX_DEBUG_LOGGED_MESSAGES={} MAX_DEBUG_GROUP_STACK_DEPTH={} "
                                     "MAX_LABEL_LENGTH={}\n",
                                     i.max_message_length, i.max_logged, i.max_depth, i.max_label);
        }

        // The state that the very first messages saw, before any glDebugMessageControl call (section 18.1).
        void area_defaults(demo::Check& check)
        {
            const Initial& i = initial_;
            // ids 0x700..0x703 carry HIGH, MEDIUM, LOW, NOTIFICATION. Only LOW (0x702) is initially disabled.
            const std::vector<GLuint> expected_log = {0x700, 0x701, 0x703};
            check.expect(i.log_ids == expected_log,
                         "message log: HIGH, MEDIUM and NOTIFICATION application messages are enabled by default, LOW is not (§18.1)",
                         std::format("got {} expected {}", describe_ids(i.log_ids), describe_ids(expected_log)));
            check.expect_eq(i.log_count_after_inserts, 3, "DEBUG_LOGGED_MESSAGES counted exactly the three enabled messages");
            check.expect(i.log_sources_application, "the logged messages all carry source DEBUG_SOURCE_APPLICATION");
            const std::vector<GLuint> expected_callback = {0x710, 0x711, 0x713};
            check.expect(i.callback_ids == expected_callback,
                         "callback: the same default (third-party markers; HIGH, MEDIUM, NOTIFICATION delivered, LOW not)",
                         std::format("got {} expected {}", describe_ids(i.callback_ids), describe_ids(expected_callback)));
        }

        // glDebugMessageCallback + glGetPointerv (sections 18.2, 18.9, 20.2).
        void area_pointers(demo::Check& check)
        {
            GLDEBUGPROC fn = nullptr;
            void* user = nullptr;
            callback_pointers(fn, user);
            check.expect(fn == &debug_callback_a, "GetPointerv(DEBUG_CALLBACK_FUNCTION) returns the registered function");
            check.expect(user == &sink_, "GetPointerv(DEBUG_CALLBACK_USER_PARAM) returns the registered user pointer");

            insert_z(Src::Application, Typ::Marker, 1, Sev::High, "via A");
            std::vector<Message> got = sink_.take();
            check.expect(got.size() == 1 && got[0].via == 1 && got[0].user == &sink_,
                         "the callback receives the userParam that was registered (it is the pointer to the sink)",
                         describe(got));

            // Only one callback exists: a second registration replaces the first, with its own user pointer.
            mgl::glDebugMessageCallback(&debug_callback_b, &sink_b_);
            callback_pointers(fn, user);
            check.expect(fn == &debug_callback_b && user == &sink_b_,
                         "a second glDebugMessageCallback replaces function and user pointer (§18.2: further calls overwrite)");
            insert_z(Src::Application, Typ::Marker, 2, Sev::High, "via B");
            check.expect(sink_.size() == 0 && sink_b_.size() == 1, "after the replacement only callback B is called, with its own user pointer",
                         std::format("A saw {} message(s), B saw {}", sink_.size(), sink_b_.size()));
            got = sink_b_.take();
            check.expect(got.size() == 1 && got[0].via == 2 && got[0].user == &sink_b_ && got[0].text == "via B",
                         "callback B received the message with userParam = &sink_b", describe(got));

            // NULL clears the callback; the message goes to the log instead (section 18.3).
            mgl::glDebugMessageCallback(nullptr, nullptr);
            callback_pointers(fn, user);
            check.expect(fn == nullptr, "glDebugMessageCallback(NULL, ...) clears DEBUG_CALLBACK_FUNCTION (§18.2)");
            insert_z(Src::Application, Typ::Marker, 3, Sev::High, "to the log");
            check.expect(sink_.size() == 0 && sink_b_.size() == 0, "with no callback neither callback is called");
            check.expect_eq(geti(mgl::GetParameter::DebugLoggedMessages), 1,
                            "with DEBUG_CALLBACK_FUNCTION NULL the message is stored in the log (DEBUG_LOGGED_MESSAGES = 1)");
            drain_log();

            // And a callback can be installed again.
            mgl::glDebugMessageCallback(&debug_callback_a, &sink_);
            callback_pointers(fn, user);
            check.expect(fn == &debug_callback_a && user == &sink_, "the callback can be registered again");
            insert_z(Src::Application, Typ::Marker, 4, Sev::High, "via A again");
            got = sink_.take();
            check.expect(got.size() == 1 && got[0].via == 1, "...and receives messages again", describe(got));
        }

        // glDebugMessageInsert as seen by the callback (sections 18.2, 18.5).
        void area_insert(demo::Check& check)
        {
            insert_z(Src::Application, Typ::Marker, 0x1234, Sev::Medium, "Hello debug output");
            std::vector<Message> got = sink_.take();
            expect_one(check, got, "Insert, null-terminated string: exact source, type, id, severity, length and text",
                       Src::Application, Typ::Marker, 0x1234, Sev::Medium, "Hello debug output");
            check.expect(!got.empty() && got[0].user == &sink_ && got[0].thread == std::this_thread::get_id(),
                         "with synchronous output the callback runs on this thread, before the insert returns (§18.8), with the user pointer");

            mgl::glDebugMessageInsert(Src::ThirdParty, Typ::Other, 5, Sev::Low, 8, "truncate me please");
            got = sink_.take();
            expect_one(check, got, "Insert with an explicit length of 8: the callback gets length = 8 and the first 8 characters",
                       Src::ThirdParty, Typ::Other, 5, Sev::Low, "truncate", false);
            expect_terminated_at_length(check, got, "...and the message string is null-terminated at that length (§18.1, §18.2)");

            insert_z(Src::Application, Typ::Other, 0, Sev::Notification, "id zero");
            insert_z(Src::Application, Typ::Other, 0xFFFFFFFFu, Sev::Notification, "id max");
            got = sink_.take();
            check.expect(got.size() == 2 && matches(got[0], Src::Application, Typ::Other, 0, Sev::Notification, "id zero") &&
                             matches(got[1], Src::Application, Typ::Other, 0xFFFFFFFFu, Sev::Notification, "id max"),
                         "ids 0 and 0xFFFFFFFF are delivered unchanged, in order", describe(got));

            // Every valid (source, type, severity) triple of section 18.5: 2 sources x 9 types x 4 severities.
            const std::array<Src, 2> sources = {Src::Application, Src::ThirdParty};
            const std::array<Typ, 9> types = {Typ::Error,       Typ::DeprecatedBehavior, Typ::UndefinedBehavior,
                                              Typ::Portability, Typ::Performance,        Typ::Other,
                                              Typ::Marker,      Typ::PushGroup,          Typ::PopGroup};
            const std::array<Sev, 4> severities = {Sev::High, Sev::Medium, Sev::Low, Sev::Notification};
            std::vector<Message> expected;
            GLuint id = 1000;
            for (Src s : sources)
                for (Typ t : types)
                    for (Sev v : severities)
                    {
                        const std::string text = std::format("combination {}", id);
                        insert_n(s, t, id, v, text);
                        Message m;
                        m.source = s;
                        m.type = t;
                        m.id = id;
                        m.severity = v;
                        m.text = text;
                        expected.push_back(m);
                        ++id;
                    }
            check.expect_no_gl_error("all 72 valid (source, type, severity) combinations are accepted without error");
            got = sink_.take();
            bool all = got.size() == expected.size();
            std::string first_bad;
            for (std::size_t k = 0; all && k < got.size(); ++k)
                if (!matches(got[k], expected[k].source, expected[k].type, expected[k].id, expected[k].severity, expected[k].text))
                {
                    all = false;
                    first_bad = std::format("#{}: got {}", k, describe(got[k]));
                }
            check.expect(all, "all 72 combinations arrive in order with their exact fields",
                         all ? "" : (first_bad.empty() ? std::format("{} of {} arrived", got.size(), expected.size()) : first_bad));
            check.expect_eq(geti(mgl::GetParameter::DebugLoggedMessages), 0,
                            "while a callback is set no message is stored in the log (§18.3)");
        }

        // Defined errors of glDebugMessageInsert (section 18.5).
        void area_insert_errors(demo::Check& check)
        {
            const std::array<Src, 5> bad_sources = {Src::Api, Src::WindowSystem, Src::ShaderCompiler, Src::Other, Src::DontCare};
            const std::array<const char*, 5> bad_source_names = {"API", "WINDOW_SYSTEM", "SHADER_COMPILER", "OTHER", "DONT_CARE"};
            for (std::size_t i = 0; i < bad_sources.size(); ++i)
            {
                insert_z(bad_sources[i], Typ::Marker, 1, Sev::High, "x");
                check.expect_gl_error(EC::InvalidEnum, std::format("Insert with source {} -> INVALID_ENUM (only APPLICATION and THIRD_PARTY)", bad_source_names[i]));
            }
            insert_z(Src::Application, Typ::DontCare, 1, Sev::High, "x");
            check.expect_gl_error(EC::InvalidEnum, "Insert with type DONT_CARE (not in Table 18.2) -> INVALID_ENUM");
            insert_z(Src::Application, Typ::Marker, 1, Sev::DontCare, "x");
            check.expect_gl_error(EC::InvalidEnum, "Insert with severity DONT_CARE (not in Table 18.3) -> INVALID_ENUM");
            check.expect_eq(sink_.size(), std::size_t{0}, "none of the rejected inserts produced a message");

            // MAX_DEBUG_MESSAGE_LENGTH: the string must be strictly shorter (section 18.5).
            const GLsizei max = initial_.max_message_length;
            const std::string too_long(static_cast<std::size_t>(max), 'm');
            insert_z(Src::Application, Typ::Marker, 2, Sev::High, too_long.c_str());
            check.expect_gl_error(EC::InvalidValue, "Insert of a null-terminated string with MAX_DEBUG_MESSAGE_LENGTH characters -> INVALID_VALUE");
            mgl::glDebugMessageInsert(Src::Application, Typ::Marker, 2, Sev::High, max, too_long.data());
            check.expect_gl_error(EC::InvalidValue, "Insert with length = MAX_DEBUG_MESSAGE_LENGTH -> INVALID_VALUE");
            check.expect_eq(sink_.size(), std::size_t{0}, "the overlong inserts produced no message");

            const std::string fits(static_cast<std::size_t>(max - 1), 'f');
            insert_z(Src::Application, Typ::Marker, 3, Sev::High, fits.c_str());
            std::vector<Message> got = sink_.take();
            expect_one(check, got, "a string of MAX_DEBUG_MESSAGE_LENGTH - 1 characters is accepted and delivered unchanged",
                       Src::Application, Typ::Marker, 3, Sev::High, fits);
            check.expect(!got.empty() && got[0].length + 1 <= max,
                         "...its length including the terminator does not exceed MAX_DEBUG_MESSAGE_LENGTH (§18.1)");
        }

        // DEBUG_OUTPUT and DEBUG_OUTPUT_SYNCHRONOUS as state (sections 18, 18.3, 18.4, 18.5, 18.8).
        void area_output_enable(demo::Check& check)
        {
            check.expect(mgl::glIsEnabled(mgl::Capability::DebugOutput), "IsEnabled(DEBUG_OUTPUT) is TRUE after Enable");
            mgl::glDisable(mgl::Capability::DebugOutput);
            check.expect(!mgl::glIsEnabled(mgl::Capability::DebugOutput), "IsEnabled(DEBUG_OUTPUT) is FALSE after Disable");

            insert_z(Src::Application, Typ::Marker, 1, Sev::High, "discarded");
            check.expect_gl_error(EC::NoError, "Insert with DEBUG_OUTPUT disabled is discarded without an error (§18.5)");
            check.expect_eq(sink_.size(), std::size_t{0}, "with DEBUG_OUTPUT disabled the callback is not called (§18.2)");
            mgl::glDebugMessageCallback(nullptr, nullptr);
            insert_z(Src::Application, Typ::Marker, 1, Sev::High, "discarded");
            check.expect_eq(geti(mgl::GetParameter::DebugLoggedMessages), 0, "with DEBUG_OUTPUT disabled no message is added to the log (§18.3)");

            mgl::glEnable(mgl::Capability::DebugOutput);
            check.expect(mgl::glIsEnabled(mgl::Capability::DebugOutput), "IsEnabled(DEBUG_OUTPUT) is TRUE again after Enable");
            insert_z(Src::Application, Typ::Marker, 2, Sev::High, "logged");
            check.expect_eq(geti(mgl::GetParameter::DebugLoggedMessages), 1,
                            "after enabling DEBUG_OUTPUT again, messages are stored in the log (callback is NULL)");
            drain_log();
            mgl::glDebugMessageCallback(&debug_callback_a, &sink_);
            insert_z(Src::Application, Typ::Marker, 3, Sev::High, "delivered");
            check.expect_eq(sink_.size(), std::size_t{1}, "...and delivered to a callback again");
            (void)sink_.take();

            // DEBUG_OUTPUT_SYNCHRONOUS (section 18.8).
            check.expect(mgl::glIsEnabled(mgl::Capability::DebugOutputSynchronous), "IsEnabled(DEBUG_OUTPUT_SYNCHRONOUS) is TRUE after Enable");
            insert_z(Src::Application, Typ::Marker, 4, Sev::High, "sync");
            const std::vector<Message> got = sink_.take();
            check.expect(got.size() == 1 && got[0].thread == std::this_thread::get_id(),
                         "synchronous output: the callback has run on the calling thread by the time the GL command returns (§18.8)");
            mgl::glDisable(mgl::Capability::DebugOutputSynchronous);
            check.expect(!mgl::glIsEnabled(mgl::Capability::DebugOutputSynchronous), "IsEnabled(DEBUG_OUTPUT_SYNCHRONOUS) is FALSE after Disable");
            mgl::glEnable(mgl::Capability::DebugOutputSynchronous);
            check.expect(mgl::glIsEnabled(mgl::Capability::DebugOutputSynchronous), "...and TRUE again after Enable");
        }

        // The model of section 18.4: one enabled flag per (source, type, severity) message, changed by every
        // glDebugMessageControl that references it.
        struct ModelEntry
        {
            Src source;
            Typ type;
            Sev severity;
            GLuint id;
            bool enabled = true;
        };

        static bool wildcard(Src filter, Src v) { return filter == Src::DontCare || filter == v; }
        static bool wildcard(Typ filter, Typ v) { return filter == Typ::DontCare || filter == v; }
        static bool wildcard(Sev filter, Sev v) { return filter == Sev::DontCare || filter == v; }

        // glDebugMessageControl by source / type / severity (section 18.4).
        void area_control_filters(demo::Check& check)
        {
            const std::array<Src, 2> sources = {Src::Application, Src::ThirdParty};
            const std::array<Typ, 4> types = {Typ::Error, Typ::Performance, Typ::Marker, Typ::Other};
            const std::array<Sev, 4> severities = {Sev::High, Sev::Medium, Sev::Low, Sev::Notification};
            std::vector<ModelEntry> model;
            GLuint id = 1;
            for (Src s : sources)
                for (Typ t : types)
                    for (Sev v : severities) model.push_back({s, t, v, id++, true});

            struct Step
            {
                Src source;
                Typ type;
                Sev severity;
                bool enable;
                const char* what;
            };
            const std::array<Step, 9> steps = {{
                {Src::DontCare, Typ::DontCare, Sev::DontCare, false, "disable everything"},
                {Src::Application, Typ::DontCare, Sev::DontCare, true, "enable all APPLICATION messages"},
                {Src::DontCare, Typ::DontCare, Sev::Low, false, "disable severity LOW of every source"},
                {Src::ThirdParty, Typ::Marker, Sev::DontCare, true, "enable THIRD_PARTY markers (all severities)"},
                {Src::DontCare, Typ::Error, Sev::High, false, "disable HIGH errors of every source"},
                {Src::Application, Typ::Performance, Sev::Notification, true, "enable APPLICATION PERFORMANCE NOTIFICATION"},
                {Src::DontCare, Typ::Marker, Sev::DontCare, false, "disable every marker"},
                {Src::ThirdParty, Typ::DontCare, Sev::Medium, true, "enable THIRD_PARTY MEDIUM of every type"},
                {Src::DontCare, Typ::DontCare, Sev::DontCare, true, "enable everything again"},
            }};

            // The initial state of the model is the baseline of reset_state(): every application and third-party
            // message is enabled.
            auto send_all = [&] {
                sink_.clear();
                for (const ModelEntry& e : model) insert_n(e.source, e.type, e.id, e.severity, std::format("message {}", e.id));
                return ids_of(sink_.take());
            };
            const std::vector<GLuint> baseline = send_all();
            check.expect_eq(baseline.size(), model.size(), "baseline: all 32 messages (2 sources x 4 types x 4 severities) are delivered");

            std::size_t smallest = model.size();
            std::size_t largest = 0;
            std::vector<GLuint> last_actual;
            std::vector<GLuint> last_expected;
            for (const Step& st : steps)
            {
                mgl::glDebugMessageControl(st.source, st.type, st.severity, 0, nullptr, st.enable ? GL_TRUE : GL_FALSE);
                for (ModelEntry& e : model)
                    if (wildcard(st.source, e.source) && wildcard(st.type, e.type) && wildcard(st.severity, e.severity))
                        e.enabled = st.enable;
                std::vector<GLuint> expected;
                for (const ModelEntry& e : model)
                    if (e.enabled) expected.push_back(e.id);
                const std::vector<GLuint> actual = send_all();
                check.expect(actual == expected, std::format("Control: {} -> {} messages delivered, as the CPU model of §18.4 predicts", st.what, expected.size()),
                             actual == expected ? std::string{} : std::format("got {} expected {}", describe_ids(actual), describe_ids(expected)));
                smallest = std::min(smallest, expected.size());
                largest = std::max(largest, expected.size());
                last_actual = actual;
                last_expected = expected;
            }
            check.expect(smallest == 0 && largest == model.size(),
                         "the scenario ran from no message to all 32 messages (the filters take effect in both directions)",
                         std::format("min={} max={}", smallest, largest));
            check.expect_no_gl_error("every valid Control call above raised no error");
        }

        // glDebugMessageControl with id lists (section 18.4).
        void area_control_ids(demo::Check& check)
        {
            struct Probe
            {
                std::vector<GLuint> app_marker, app_other, tp_marker;
            };
            auto probe = [&] {
                sink_.clear();
                for (GLuint id = 1; id <= 6; ++id)
                {
                    insert_z(Src::Application, Typ::Marker, id, Sev::High, "id probe");
                    insert_z(Src::Application, Typ::Other, id, Sev::High, "id probe");
                    insert_z(Src::ThirdParty, Typ::Marker, id, Sev::High, "id probe");
                }
                const std::vector<Message> got = sink_.take();
                return Probe{ids_of(got, Src::Application, Typ::Marker), ids_of(got, Src::Application, Typ::Other),
                             ids_of(got, Src::ThirdParty, Typ::Marker)};
            };
            const std::vector<GLuint> all = {1, 2, 3, 4, 5, 6};

            Probe p = probe();
            check.expect(p.app_marker == all && p.app_other == all && p.tp_marker == all,
                         "baseline: ids 1..6 of three (source, type) namespaces are all delivered");

            const std::array<GLuint, 2> off = {2, 4};
            mgl::glDebugMessageControl(Src::Application, Typ::Marker, Sev::DontCare, off, GL_FALSE);
            p = probe();
            const std::vector<GLuint> without_2_4 = {1, 3, 5, 6};
            check.expect(p.app_marker == without_2_4,
                         "Control(APPLICATION, MARKER, ids {2,4}, FALSE): those two ids are no longer delivered",
                         std::format("got {}", describe_ids(p.app_marker)));
            check.expect(p.app_other == all && p.tp_marker == all,
                         "...other (source, type) pairs are separate namespaces with the same ids: unaffected (§18.1)",
                         std::format("APPLICATION/OTHER {} THIRD_PARTY/MARKER {}", describe_ids(p.app_other), describe_ids(p.tp_marker)));
            // The negative control: an expectation that ignored the Control call (all six ids) is contradicted.
            check.expect(p.app_marker != all, "negative control: the expectation 'Control had no effect' would fail");

            check.expect(!delivered(Src::Application, Typ::Marker, 2, Sev::Low),
                         "the id filter does not depend on severity: id 2 is also blocked at severity LOW");

            const std::array<GLuint, 2> on = {4, 99};
            mgl::glDebugMessageControl(Src::Application, Typ::Marker, Sev::DontCare, on, GL_TRUE);
            check.expect_gl_error(EC::NoError, "a list with a never-used id (99) raises no error: unrecognized ids are ignored (§18.4)");
            p = probe();
            const std::vector<GLuint> without_2 = {1, 3, 4, 5, 6};
            check.expect(p.app_marker == without_2, "Control(..., ids {4,99}, TRUE) re-enables id 4 only", std::format("got {}", describe_ids(p.app_marker)));
            check.expect(delivered(Src::Application, Typ::Marker, 99, Sev::High), "id 99 (enabled while unknown) is delivered");

            // The enabled state is stored per message: a later severity-wide enable overrides the earlier id disable.
            mgl::glDebugMessageControl(Src::DontCare, Typ::DontCare, Sev::High, 0, nullptr, GL_TRUE);
            p = probe();
            check.expect(p.app_marker == all, "Control(DONT_CARE, DONT_CARE, HIGH, TRUE) re-enables id 2 as well: the state is per message (§18.4)",
                         std::format("got {}", describe_ids(p.app_marker)));

            // With count == 0 the ids are ignored and the call refers to every message that matches the other filters.
            const std::array<GLuint, 2> ignored = {1, 3};
            mgl::glDebugMessageControl(Src::Application, Typ::Marker, Sev::DontCare, 0, ignored.data(), GL_FALSE);
            p = probe();
            check.expect(p.app_marker.empty() && p.app_other == all && p.tp_marker == all,
                         "count = 0: ids are ignored, so Control(APPLICATION, MARKER, ..., FALSE) disables every APPLICATION marker",
                         std::format("APPLICATION/MARKER {}", describe_ids(p.app_marker)));
        }

        // Defined errors of glDebugMessageControl (section 18.4).
        void area_control_errors(demo::Check& check)
        {
            const std::array<GLuint, 2> ids = {1, 2};
            mgl::glDebugMessageControl(Src::DontCare, Typ::Marker, Sev::DontCare, 1, ids.data(), GL_FALSE);
            check.expect_gl_error(EC::InvalidOperation, "Control with count > 0 and source DONT_CARE -> INVALID_OPERATION");
            mgl::glDebugMessageControl(Src::Application, Typ::DontCare, Sev::DontCare, 1, ids.data(), GL_FALSE);
            check.expect_gl_error(EC::InvalidOperation, "Control with count > 0 and type DONT_CARE -> INVALID_OPERATION");
            mgl::glDebugMessageControl(Src::Application, Typ::Marker, Sev::High, 1, ids.data(), GL_FALSE);
            check.expect_gl_error(EC::InvalidOperation, "Control with count > 0 and severity != DONT_CARE -> INVALID_OPERATION");
            mgl::glDebugMessageControl(Src::Application, Typ::Marker, Sev::DontCare, -1, ids.data(), GL_FALSE);
            check.expect_gl_error(EC::InvalidValue, "Control with a negative count -> INVALID_VALUE");

            check.expect(delivered(Src::Application, Typ::Marker, 1, Sev::High) && delivered(Src::Application, Typ::Marker, 2, Sev::High),
                         "none of the failed calls changed any enabled state: ids 1 and 2 are still delivered");

            mgl::glDebugMessageControl(Src::DontCare, Typ::DontCare, Sev::DontCare, 0, ids.data(), GL_TRUE);
            check.expect_gl_error(EC::NoError, "count = 0 ignores ids: DONT_CARE for all three filters is valid");
            mgl::glDebugMessageControl(Src::Application, Typ::Marker, Sev::DontCare, static_cast<GLsizei>(ids.size()), ids.data(), GL_FALSE);
            check.expect_gl_error(EC::NoError, "count > 0 with concrete source, concrete type and severity DONT_CARE is valid");
            check.expect(!delivered(Src::Application, Typ::Marker, 1, Sev::High), "...and takes effect");
        }

        // The message log (sections 18.3, 18.9).
        void area_log(demo::Check& check)
        {
            mgl::glDebugMessageCallback(nullptr, nullptr);
            GLDEBUGPROC fn = &debug_callback_a;
            void* user = nullptr;
            callback_pointers(fn, user);
            check.expect(fn == nullptr, "the callback is NULL: messages are stored in the log (§18.3)");
            check.expect_eq(geti(mgl::GetParameter::DebugLoggedMessages), 0, "the log starts empty");
            check.expect_eq(geti(mgl::GetParameter::DebugNextLoggedMessageLength), 0, "DEBUG_NEXT_LOGGED_MESSAGE_LENGTH is 0 for an empty log");

            insert_z(Src::Application, Typ::Marker, 11, Sev::High, "first");
            insert_n(Src::ThirdParty, Typ::Performance, 12, Sev::Medium, "second");
            insert_z(Src::Application, Typ::Other, 13, Sev::Low, "third message");
            insert_z(Src::ThirdParty, Typ::Error, 14, Sev::Notification, "4th");
            check.expect_eq(geti(mgl::GetParameter::DebugLoggedMessages), 4, "DEBUG_LOGGED_MESSAGES counts the four messages");
            check.expect_eq(geti(mgl::GetParameter::DebugNextLoggedMessageLength), 6,
                            "DEBUG_NEXT_LOGGED_MESSAGE_LENGTH is the length of \"first\" including its terminator (6)");

            Fetched f = fetch_log(2, 256);
            check.expect_eq(f.count, 2u, "GetDebugMessageLog(count = 2) returns 2 messages");
            check.expect(f.ids[0] == 11 && f.ids[1] == 12 && f.sources[0] == Src::Application && f.sources[1] == Src::ThirdParty &&
                             f.types[0] == Typ::Marker && f.types[1] == Typ::Performance && f.severities[0] == Sev::High &&
                             f.severities[1] == Sev::Medium,
                         "...the oldest two, in order, with their source, type, id and severity");
            check.expect(f.lengths[0] == 6 && f.lengths[1] == 7, "the returned lengths include the null terminators (6 and 7)",
                         std::format("{} and {}", f.lengths[0], f.lengths[1]));
            check.expect(std::string_view(f.text.data(), 13) == std::string_view("first\0second\0", 13),
                         "the strings are concatenated, each followed by a null terminator: \"first\\0second\\0\"");
            check.expect_eq(geti(mgl::GetParameter::DebugLoggedMessages), 2, "fetched messages are removed from the log");
            check.expect_eq(geti(mgl::GetParameter::DebugNextLoggedMessageLength), 14,
                            "DEBUG_NEXT_LOGGED_MESSAGE_LENGTH now describes \"third message\" (14 with terminator)");

            // bufSize: a message that does not fit completely is not fetched, and neither are the ones after it.
            f = fetch_log(8, 14 + 3);  // "third message" (14) fits; "4th" (4) would need 18
            check.expect(f.count == 1 && f.ids[0] == 13 && f.lengths[0] == 14,
                         "bufSize = 17: \"third message\" (14) fits, \"4th\" (4) does not -> exactly one message fetched",
                         std::format("count={}", f.count));
            check.expect_eq(geti(mgl::GetParameter::DebugLoggedMessages), 1, "the message that did not fit stays in the log");
            f = fetch_log(8, 3);  // "4th" needs 4 bytes
            check.expect_eq(f.count, 0u, "bufSize = 3 is too small for \"4th\" (4 with terminator): nothing is fetched");
            check.expect_eq(geti(mgl::GetParameter::DebugLoggedMessages), 1, "...and the message remains in the log");
            f = fetch_log(0, 256);
            check.expect(f.count == 0 && geti(mgl::GetParameter::DebugLoggedMessages) == 1, "count = 0 fetches nothing and removes nothing");

            // Null arrays: the attributes are discarded but the messages are still removed.
            const GLuint removed = mgl::glGetDebugMessageLog(8, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
            check.expect(removed == 1 && geti(mgl::GetParameter::DebugLoggedMessages) == 0,
                         "all arrays NULL: the message is removed and counted, its attributes are discarded (§18.9)",
                         std::format("returned {}", removed));
            check.expect_eq(geti(mgl::GetParameter::DebugNextLoggedMessageLength), 0, "an empty log again reports length 0");

            // Each array is optional on its own.
            insert_z(Src::Application, Typ::Marker, 21, Sev::High, "alpha");
            insert_z(Src::Application, Typ::Marker, 22, Sev::High, "beta");
            std::array<GLuint, 2> only_ids = {0, 0};
            const GLuint n = mgl::glGetDebugMessageLog(2, 0, nullptr, nullptr, only_ids.data(), nullptr, nullptr, nullptr);
            check.expect(n == 2 && only_ids[0] == 21 && only_ids[1] == 22 && geti(mgl::GetParameter::DebugLoggedMessages) == 0,
                         "only the ids array is requested: ids come back, the strings are skipped, the messages are removed",
                         std::format("returned {}", n));

            // Errors.
            insert_z(Src::Application, Typ::Marker, 23, Sev::High, "gamma");
            std::array<char, 16> text{};
            text.fill('#');
            const GLuint failed = mgl::glGetDebugMessageLog(1, -1, nullptr, nullptr, nullptr, nullptr, nullptr, text.data());
            check.expect_gl_error(EC::InvalidValue, "GetDebugMessageLog with a negative bufSize and a non-NULL messageLog -> INVALID_VALUE");
            check.expect(failed == 0 && geti(mgl::GetParameter::DebugLoggedMessages) == 1 && text[0] == '#',
                         "...the failed call returns 0, fetches nothing and writes nothing (§2.3.1: the command is ignored)",
                         std::format("returned {}, DEBUG_LOGGED_MESSAGES = {}", failed, geti(mgl::GetParameter::DebugLoggedMessages)));
            (void)mgl::glGetDebugMessageLog(1, -1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
            check.expect_gl_error(EC::NoError, "...but a negative bufSize with a NULL messageLog is not an error (§18.9)");
            drain_log();

            // A callback and the log are exclusive (section 18.3).
            mgl::glDebugMessageCallback(&debug_callback_a, &sink_);
            insert_z(Src::Application, Typ::Marker, 31, Sev::High, "callback");
            check.expect(sink_.size() == 1 && geti(mgl::GetParameter::DebugLoggedMessages) == 0,
                         "with a callback set the message reaches the callback and is not stored in the log (§18.3)");
        }

        // MAX_DEBUG_LOGGED_MESSAGES (section 18.3).
        void area_log_overflow(demo::Check& check)
        {
            mgl::glDebugMessageCallback(nullptr, nullptr);
            const GLint max = initial_.max_logged;
            const GLint total = max + 5;
            for (GLint i = 0; i < total; ++i)
                insert_z(Src::Application, Typ::Marker, 100 + static_cast<GLuint>(i), Sev::Notification, "overflow");
            check.expect_eq(geti(mgl::GetParameter::DebugLoggedMessages), max,
                            "inserting MAX_DEBUG_LOGGED_MESSAGES + 5 messages leaves exactly MAX_DEBUG_LOGGED_MESSAGES in the log");
            const Fetched f = fetch_log(static_cast<GLuint>(total), total * 16);
            std::vector<GLuint> expected;
            for (GLint i = 0; i < max; ++i) expected.push_back(100 + static_cast<GLuint>(i));
            const std::vector<GLuint> got(f.ids.begin(), f.ids.begin() + f.count);
            check.expect(got == expected,
                         "the log kept the OLDEST messages; the later ones were discarded while it was full (§18.3)",
                         std::format("got {} expected {}", describe_ids(got), describe_ids(expected)));
            check.expect_eq(geti(mgl::GetParameter::DebugLoggedMessages), 0, "after fetching, the log is empty");
            insert_z(Src::Application, Typ::Marker, 999, Sev::Notification, "after");
            check.expect_eq(geti(mgl::GetParameter::DebugLoggedMessages), 1, "once the log was cleared, new messages are stored again");
            drain_log();
        }

        // Debug groups, part 1: the messages (section 18.6).
        void area_group_messages(demo::Check& check)
        {
            check.expect_eq(geti(mgl::GetParameter::DebugGroupStackDepth), 1, "DEBUG_GROUP_STACK_DEPTH is 1 at the start");
            mgl::glPushDebugGroup(Src::Application, 77, -1, "outer");
            std::vector<Message> got = sink_.take();
            expect_one(check, got, "PushDebugGroup generates a PUSH_GROUP message with the given source and id, severity NOTIFICATION",
                       Src::Application, Typ::PushGroup, 77, Sev::Notification, "outer");
            check.expect_eq(geti(mgl::GetParameter::DebugGroupStackDepth), 2, "DEBUG_GROUP_STACK_DEPTH is 2 after one push");

            mgl::glPushDebugGroup(Src::ThirdParty, 88, 5, "inner-and-more");
            got = sink_.take();
            expect_one(check, got, "PushDebugGroup with length 5 uses only 5 characters (THIRD_PARTY source)",
                       Src::ThirdParty, Typ::PushGroup, 88, Sev::Notification, "inner", false);
            expect_terminated_at_length(check, got, "...and the PUSH_GROUP message string is null-terminated at that length");
            check.expect_eq(geti(mgl::GetParameter::DebugGroupStackDepth), 3, "DEBUG_GROUP_STACK_DEPTH is 3 after two pushes");

            insert_z(Src::Application, Typ::Marker, 5, Sev::High, "inside the group");
            got = sink_.take();
            expect_one(check, got, "a message inserted inside a group is delivered normally", Src::Application, Typ::Marker, 5, Sev::High,
                       "inside the group");

            mgl::glPopDebugGroup();
            got = sink_.take();
            const bool inner_pop = got.size() == 1 && got[0].source == Src::ThirdParty && got[0].type == Typ::PopGroup && got[0].id == 88 &&
                                   got[0].severity == Sev::Notification && got[0].terminated;
            check.expect(inner_pop, "PopDebugGroup generates a POP_GROUP message with the source and id of the matching push, severity NOTIFICATION",
                         describe(got));
            if (got.size() == 1) std::cout << std::format("   note: the POP_GROUP message text is '{}' (implementation-defined, \"based on\" the push message)\n", got[0].text);
            check.expect_eq(geti(mgl::GetParameter::DebugGroupStackDepth), 2, "DEBUG_GROUP_STACK_DEPTH is 2 after one pop");

            mgl::glPopDebugGroup();
            got = sink_.take();
            check.expect(got.size() == 1 && got[0].source == Src::Application && got[0].type == Typ::PopGroup && got[0].id == 77 &&
                             got[0].severity == Sev::Notification,
                         "the second pop reports the OUTER group (APPLICATION, id 77): groups are a stack", describe(got));
            check.expect_eq(geti(mgl::GetParameter::DebugGroupStackDepth), 1, "back to the default group: DEBUG_GROUP_STACK_DEPTH is 1");

            // Errors of PushDebugGroup.
            const std::array<Src, 4> bad = {Src::Api, Src::WindowSystem, Src::Other, Src::DontCare};
            const std::array<const char*, 4> names = {"API", "WINDOW_SYSTEM", "OTHER", "DONT_CARE"};
            for (std::size_t i = 0; i < bad.size(); ++i)
            {
                mgl::glPushDebugGroup(bad[i], 1, -1, "x");
                check.expect_gl_error(EC::InvalidEnum, std::format("PushDebugGroup with source {} -> INVALID_ENUM", names[i]));
            }
            const std::string too_long(static_cast<std::size_t>(initial_.max_message_length), 'g');
            mgl::glPushDebugGroup(Src::Application, 1, -1, too_long.c_str());
            check.expect_gl_error(EC::InvalidValue,
                                  "PushDebugGroup with length < 0 and MAX_DEBUG_MESSAGE_LENGTH characters -> INVALID_VALUE");
            check.expect(geti(mgl::GetParameter::DebugGroupStackDepth) == 1 && sink_.size() == 0,
                         "the failed pushes changed no depth and generated no message");
            const std::string fits(static_cast<std::size_t>(initial_.max_message_length - 1), 'h');
            mgl::glPushDebugGroup(Src::Application, 2, -1, fits.c_str());
            got = sink_.take();
            expect_one(check, got, "a group message of MAX_DEBUG_MESSAGE_LENGTH - 1 characters is accepted", Src::Application,
                       Typ::PushGroup, 2, Sev::Notification, fits);
            mgl::glPopDebugGroup();
            (void)sink_.take();
        }

        // Debug groups, part 2: the filter state belongs to the group (sections 18.4, 18.6).
        void area_group_filters(demo::Check& check)
        {
            const std::array<GLuint, 1> one = {1};
            mgl::glDebugMessageControl(Src::Application, Typ::Marker, Sev::DontCare, one, GL_FALSE);
            check.expect(!delivered(Src::Application, Typ::Marker, 1, Sev::High) && delivered(Src::Application, Typ::Marker, 2, Sev::High),
                         "default group: marker id 1 disabled, id 2 enabled");

            mgl::glPushDebugGroup(Src::Application, 500, -1, "g1");
            check.expect(!delivered(Src::Application, Typ::Marker, 1, Sev::High),
                         "a new group inherits the filter state of its parent: id 1 is still disabled in g1 (§18.6)");
            mgl::glDebugMessageControl(Src::Application, Typ::Marker, Sev::DontCare, one, GL_TRUE);
            mgl::glDebugMessageControl(Src::Application, Typ::Marker, Sev::Low, 0, nullptr, GL_FALSE);
            check.expect(delivered(Src::Application, Typ::Marker, 1, Sev::High) && !delivered(Src::Application, Typ::Marker, 2, Sev::Low) &&
                             delivered(Src::Application, Typ::Marker, 2, Sev::High),
                         "g1: after Control, id 1 is enabled and LOW markers are disabled");

            mgl::glPushDebugGroup(Src::Application, 501, -1, "g2");
            check.expect(delivered(Src::Application, Typ::Marker, 1, Sev::High) && !delivered(Src::Application, Typ::Marker, 2, Sev::Low),
                         "g2 (nested) inherits g1's state, not the default group's");
            mgl::glDebugMessageControl(Src::Application, Typ::Marker, Sev::DontCare, one, GL_FALSE);
            check.expect(!delivered(Src::Application, Typ::Marker, 1, Sev::High), "g2: id 1 disabled again");
            check.expect_eq(geti(mgl::GetParameter::DebugGroupStackDepth), 3, "DEBUG_GROUP_STACK_DEPTH is 3 inside g2");

            mgl::glPopDebugGroup();
            check.expect(delivered(Src::Application, Typ::Marker, 1, Sev::High) && !delivered(Src::Application, Typ::Marker, 2, Sev::Low),
                         "after popping g2 the state of g1 is back: id 1 enabled, LOW disabled (§18.6)");

            mgl::glPopDebugGroup();
            check.expect(!delivered(Src::Application, Typ::Marker, 1, Sev::High) && delivered(Src::Application, Typ::Marker, 2, Sev::Low),
                         "after popping g1 the default group's state is back: id 1 disabled, LOW markers enabled again");
            check.expect_eq(geti(mgl::GetParameter::DebugGroupStackDepth), 1, "DEBUG_GROUP_STACK_DEPTH is 1 again");
        }

        // STACK_OVERFLOW and STACK_UNDERFLOW (section 18.6).
        void area_group_limits(demo::Check& check)
        {
            const GLint max = initial_.max_depth;
            mgl::glPopDebugGroup();
            check.expect_gl_error(EC::StackUnderflow, "PopDebugGroup with only the default group on the stack -> STACK_UNDERFLOW");
            check.expect_eq(geti(mgl::GetParameter::DebugGroupStackDepth), 1, "...and the depth stays 1");
            sink_.clear();

            // The default group counts as the first stack entry: DEBUG_GROUP_STACK_DEPTH may reach
            // MAX_DEBUG_GROUP_STACK_DEPTH, i.e. MAX - 1 groups can be pushed (see the README for the reading of §18.6).
            int errors = 0;
            for (GLint i = 0; i < max - 1; ++i)
            {
                mgl::glPushDebugGroup(i % 2 == 0 ? Src::Application : Src::ThirdParty, 2000 + static_cast<GLuint>(i), -1, "level");
                if (demo::drain_gl_errors() != EC::NoError) ++errors;
            }
            check.expect_eq(errors, 0, "MAX_DEBUG_GROUP_STACK_DEPTH - 1 pushes raise no error");
            check.expect_eq(geti(mgl::GetParameter::DebugGroupStackDepth), max, "DEBUG_GROUP_STACK_DEPTH reached MAX_DEBUG_GROUP_STACK_DEPTH");
            mgl::glPushDebugGroup(Src::Application, 9999, -1, "one too many");
            check.expect_gl_error(EC::StackOverflow, "the next PushDebugGroup -> STACK_OVERFLOW");
            check.expect_eq(geti(mgl::GetParameter::DebugGroupStackDepth), max, "...and the depth stays at the maximum");

            std::vector<Message> got = only_type(sink_.take(), Typ::PushGroup);
            bool pushes_ok = static_cast<GLint>(got.size()) >= max - 1;
            for (GLint i = 0; pushes_ok && i < max - 1; ++i)
                pushes_ok = got[static_cast<std::size_t>(i)].id == 2000 + static_cast<GLuint>(i) &&
                            got[static_cast<std::size_t>(i)].source == (i % 2 == 0 ? Src::Application : Src::ThirdParty);
            check.expect(pushes_ok, "the successful pushes produced PUSH_GROUP messages with the ids and sources they were given, in order",
                         std::format("{} PUSH_GROUP messages", got.size()));

            for (GLint i = 0; i < max - 1; ++i) mgl::glPopDebugGroup();
            check.expect_gl_error(EC::NoError, "MAX_DEBUG_GROUP_STACK_DEPTH - 1 pops raise no error");
            got = only_type(sink_.take(), Typ::PopGroup);
            bool pops_ok = static_cast<GLint>(got.size()) == max - 1;
            for (GLint i = 0; pops_ok && i < max - 1; ++i)
            {
                const GLint pushed = max - 2 - i;  // LIFO: the last pushed group is popped first
                pops_ok = got[static_cast<std::size_t>(i)].id == 2000 + static_cast<GLuint>(pushed) &&
                          got[static_cast<std::size_t>(i)].source == (pushed % 2 == 0 ? Src::Application : Src::ThirdParty) &&
                          got[static_cast<std::size_t>(i)].severity == Sev::Notification;
            }
            check.expect(pops_ok, "the POP_GROUP messages come in reverse order with the id and source of their push (a stack)",
                         std::format("{} POP_GROUP messages", got.size()));
            check.expect_eq(geti(mgl::GetParameter::DebugGroupStackDepth), 1, "all groups popped: depth is 1");
            mgl::glPopDebugGroup();
            check.expect_gl_error(EC::StackUnderflow, "one more PopDebugGroup -> STACK_UNDERFLOW");
        }

        // Errors and debug output (section 2.3.1: "the GL may also generate a debug output message").
        void area_error_messages(demo::Check& check)
        {
            control_all(true);  // every source, including the GL's own API messages
            struct Case
            {
                const char* what;
                EC code;
                std::function<void()> provoke;
            };
            const GLuint never = 0x7ff00001u;
            const std::array<GLuint, 1> ids = {1};
            const std::array<Case, 4> cases = {{
                {"glObjectLabel on a buffer name that never existed", EC::InvalidValue,
                 [&] { mgl::glObjectLabel(Ns::Buffer, never, -1, "x"); }},
                {"glDebugMessageInsert with source API", EC::InvalidEnum,
                 [&] { insert_z(Src::Api, Typ::Marker, 1, Sev::High, "x"); }},
                {"glPopDebugGroup on the default group", EC::StackUnderflow, [&] { mgl::glPopDebugGroup(); }},
                {"glDebugMessageControl with ids and source DONT_CARE", EC::InvalidOperation,
                 [&] { mgl::glDebugMessageControl(Src::DontCare, Typ::Marker, Sev::DontCare, 1, ids.data(), GL_TRUE); }},
            }};
            std::size_t arrived = 0;
            for (const Case& c : cases)
            {
                sink_.clear();
                c.provoke();
                check.expect_gl_error(c.code, std::format("{}: the error code is set", c.what));
                const std::vector<Message> got = sink_.take();
                // §2.3.1 says the GL *may* generate a message; if it does, it has source API and type ERROR. The
                // ERROR-typed messages are checked, their arrival is not required.
                const std::vector<Message> errors = only_type(got, Typ::Error);
                bool shape = true;
                for (const Message& m : errors)
                    shape = shape && m.source == Src::Api && !m.text.empty() && m.terminated && m.length + 1 <= initial_.max_message_length;
                check.expect(shape, std::format("{}: every ERROR message that arrives has source API and a well-formed string (§2.3.1)", c.what),
                             shape ? std::string{} : describe(errors));
                arrived += errors.size();
                for (const Message& m : errors) std::cout << std::format("   note: driver error message {}\n", describe(m));
            }
            std::cout << std::format("   note: {} of {} provoked errors produced a debug message (allowed but not required by §2.3.1)\n", arrived, cases.size());

            // With DEBUG_OUTPUT disabled the GL generates nothing at all (section 18).
            mgl::glDisable(mgl::Capability::DebugOutput);
            sink_.clear();
            mgl::glObjectLabel(Ns::Buffer, never, -1, "x");
            check.expect_gl_error(EC::InvalidValue, "with DEBUG_OUTPUT disabled the error code is still set");
            check.expect_eq(sink_.size(), std::size_t{0}, "...but no message is generated (§18: the GL will not generate any debug output)");
            mgl::glEnable(mgl::Capability::DebugOutput);
        }

        // The scene's stream, captured during render(), and its pixels.
        void area_scene(const demo::Env& env, demo::Check& check)
        {
            std::vector<Message> app;
            std::size_t implementation = 0;
            for (const Message& m : scene_stream_)
            {
                if (m.source == Src::Application) app.push_back(m);
                else ++implementation;
            }
            const bool stream = app.size() == 4 && matches(app[0], Src::Application, Typ::PushGroup, kSceneGroupId, Sev::Notification, "Scene") &&
                                matches(app[1], Src::Application, Typ::Marker, kMarkerTriangleId, Sev::Notification, "triangle") &&
                                matches(app[2], Src::Application, Typ::Marker, kMarkerQuadId, Sev::Notification, "quad") &&
                                app[3].type == Typ::PopGroup && app[3].id == kSceneGroupId && app[3].severity == Sev::Notification;
            check.expect(stream,
                         "render() annotated the frame: PUSH_GROUP \"Scene\", markers \"triangle\" and \"quad\", POP_GROUP - delivered under the DEFAULT filter state",
                         describe(app));
            std::cout << std::format("   note: {} implementation message(s) (source != APPLICATION) arrived during the frame\n", implementation);

            // Pixels: the debug calls changed nothing in the picture.
            const int w = env.width;
            auto project = [&](float nx, float ny) {
                return std::pair<int, int>{static_cast<int>((nx + 1.0f) * 0.5f * static_cast<float>(w)),
                                           scene_y_ + static_cast<int>((ny + 1.0f) * 0.5f * static_cast<float>(scene_h_))};
            };
            const Rgba8 clear = {to8(kClear[0]), to8(kClear[1]), to8(kClear[2]), 255};
            const Rgba8 orange = {to8(kTriangleColor[0]), to8(kTriangleColor[1]), to8(kTriangleColor[2]), 255};
            // Triangle: barycentric points well inside (weights >= 0.2).
            auto tri_point = [&](float a, float b, float c) {
                const float nx = a * kVertices[0].x + b * kVertices[1].x + c * kVertices[2].x;
                const float ny = a * kVertices[0].y + b * kVertices[1].y + c * kVertices[2].y;
                return project(nx, ny);
            };
            auto p = tri_point(0.34f, 0.33f, 0.33f);
            check.expect_pixel(p.first, p.second, orange, 1, "triangle centroid shows the flat colour of u_color (layout(location = 0) uniform)");
            p = tri_point(0.4f, 0.4f, 0.2f);
            check.expect_pixel(p.first, p.second, orange, 1, "a second point inside the triangle shows the same colour");
            // Quad: the centres of the four texels of the 2x2 texture (NEAREST sampler, layout(binding = 0) sampler2D).
            const struct
            {
                float u, v;
                Rgba8 color;
                const char* name;
            } texels[] = {{0.25f, 0.25f, {255, 0, 0, 255}, "red"},
                          {0.75f, 0.25f, {0, 255, 0, 255}, "green"},
                          {0.25f, 0.75f, {0, 0, 255, 255}, "blue"},
                          {0.75f, 0.75f, {255, 255, 0, 255}, "yellow"}};
            for (const auto& t : texels)
            {
                const auto q = project(0.1f + t.u * 0.8f, -0.8f + t.v * 1.6f);
                check.expect_pixel(q.first, q.second, t.color, 1, std::format("quad texel ({}, {}): {}", t.u < 0.5f ? 0 : 1, t.v < 0.5f ? 0 : 1, t.name));
            }
            const auto bg = project(0.0f, 0.0f);
            check.expect_pixel(bg.first, bg.second, clear, 1, "between triangle and quad: the clear colour");
            check.expect_pixel(2, 2, clear, 1, "the panel region is still the clear colour at this point (drawn after the checks)");

            // Negative control: the pixel of the triangle is not what a wrong expectation (the quad's red texel) says.
            const Rgba8 got = demo::read_pixel(p.first, p.second);
            check.expect(got != Rgba8{255, 0, 0, 255}, "negative control: the triangle pixel differs from a wrong expectation (red), so the pixel checks can fail",
                         demo::format_rgba(got));
        }

        // The five scene objects still carry the labels given in init(), after being used by draw calls.
        void area_scene_labels(demo::Check& check)
        {
            const std::array<std::pair<Ns, GLuint>, 5> objects = {{{Ns::Program, program_.value},
                                                                    {Ns::VertexArray, vao_.value},
                                                                    {Ns::Buffer, vbo_.value},
                                                                    {Ns::Texture, texture_.value},
                                                                    {Ns::Sampler, sampler_.value}}};
            for (std::size_t i = 0; i < objects.size(); ++i)
            {
                const auto [ns, name] = objects[i];
                const LabelRead r = read_label([&](GLsizei b, GLsizei* l, GLchar* s) { mgl::glGetObjectLabel(ns, name, b, l, s); }, 64);
                const bool ok = label_is(r, kSceneLabels[i]);
                scene_label_ok_[i] = ok;
                check.expect(ok, std::format("'{}' keeps its label after being used by the draw calls", kSceneLabels[i]), ok ? std::string{} : describe(r));
            }
        }

        // The whole label battery for one object namespace (Table 18.4, sections 18.7, 18.9).
        void area_labels(demo::Check& check, const LabelSubject& s)
        {
            const GLuint a = s.create();
            const GLuint b = s.create();
            check.expect_no_gl_error(std::format("{}: creating two objects raises no error", s.name));
            const SetLabel set_a = [&](GLsizei length, const GLchar* label) { mgl::glObjectLabel(s.ns, a, length, label); };
            const GetLabel get_a = [&](GLsizei buf, GLsizei* length, GLchar* label) { mgl::glGetObjectLabel(s.ns, a, buf, length, label); };
            const SetLabel set_b = [&](GLsizei length, const GLchar* label) { mgl::glObjectLabel(s.ns, b, length, label); };
            const GetLabel get_b = [&](GLsizei buf, GLsizei* length, GLchar* label) { mgl::glGetObjectLabel(s.ns, b, buf, length, label); };

            label_battery(check, s.name, set_a, get_a, set_b, get_b);

            // Invalid names (section 18.7, 18.9).
            const GLuint never = std::max(a, b) + 0x10000u;
            expect_label_invalid(check, std::format("{}: a name that never existed", s.name),
                                 [&](GLsizei l, const GLchar* t) { mgl::glObjectLabel(s.ns, never, l, t); },
                                 [&](GLsizei bs, GLsizei* l, GLchar* t) { mgl::glGetObjectLabel(s.ns, never, bs, l, t); });
            s.destroy(b);
            expect_label_invalid(check, std::format("{}: a deleted object's name", s.name),
                                 [&](GLsizei l, const GLchar* t) { mgl::glObjectLabel(s.ns, b, l, t); },
                                 [&](GLsizei bs, GLsizei* l, GLchar* t) { mgl::glGetObjectLabel(s.ns, b, bs, l, t); });
            // The label of the surviving object is untouched by all of this.
            check.expect(label_is(read_label(get_a, 512), "x"), std::format("{}: the other object's label was not disturbed by the failed calls", s.name));
            s.destroy(a);
        }

        // The sync object has its own entry points: glObjectPtrLabel / glGetObjectPtrLabel.
        void area_sync_labels(demo::Check& check)
        {
            const GLsync fence_a = mgl::glFenceSync(mgl::SyncCondition::GpuCommandsComplete, mgl::SyncFlag::None);
            const GLsync fence_b = mgl::glFenceSync(mgl::SyncCondition::GpuCommandsComplete, mgl::SyncFlag::None);
            check.expect(fence_a != nullptr && fence_b != nullptr, "two fence sync objects were created");
            const SetLabel set_a = [&](GLsizei length, const GLchar* label) { mgl::glObjectPtrLabel(fence_a, length, label); };
            const GetLabel get_a = [&](GLsizei buf, GLsizei* length, GLchar* label) { mgl::glGetObjectPtrLabel(fence_a, buf, length, label); };
            const SetLabel set_b = [&](GLsizei length, const GLchar* label) { mgl::glObjectPtrLabel(fence_b, length, label); };
            const GetLabel get_b = [&](GLsizei buf, GLsizei* length, GLchar* label) { mgl::glGetObjectPtrLabel(fence_b, buf, length, label); };

            label_battery(check, "sync", set_a, get_a, set_b, get_b);

            int not_a_sync = 0;  // an address that is certainly not a sync object
            expect_label_invalid(check, "sync: a pointer that is not a sync object",
                                 [&](GLsizei l, const GLchar* t) { mgl::glObjectPtrLabel(&not_a_sync, l, t); },
                                 [&](GLsizei bs, GLsizei* l, GLchar* t) { mgl::glGetObjectPtrLabel(&not_a_sync, bs, l, t); });
            check.expect(label_is(read_label(get_a, 512), "x"), "sync: the labelled sync object was not disturbed by the failed calls");
            mgl::glFinish();
            mgl::glDeleteSync(fence_a);
            mgl::glDeleteSync(fence_b);
        }

        /// Common label semantics for any object: initial state, set/get, explicit length, NULL label, truncation,
        /// MAX_LABEL_LENGTH. `a` and `b` are two different objects of the same kind. At the end a's label is "x".
        void label_battery(demo::Check& check, const char* ns, const SetLabel& set_a, const GetLabel& get_a, const SetLabel& set_b,
                           const GetLabel& get_b)
        {
            const auto N = [&](std::string_view what) { return std::format("{}: {}", ns, what); };
            const GLsizei max = initial_.max_label;

            LabelRead r = read_label(get_a, 512);
            check.expect(label_is(r, ""), N("a new object has the empty label, length 0 (§18.7)"), describe(r));

            set_a(-1, "alpha label");
            r = read_label(get_a, 512);
            check.expect(label_is(r, "alpha label"), N("ObjectLabel with a null-terminated string; GetObjectLabel returns it with length 11"), describe(r));

            set_a(5, "abcdefgh");
            r = read_label(get_a, 512);
            check.expect(label_is(r, "abcde"), N("ObjectLabel with length 5 stores only 5 characters"), describe(r));

            LabelRead rb = read_label(get_b, 512);
            check.expect(label_is(rb, ""), N("labelling one object leaves another object's label empty"), describe(rb));
            set_b(-1, "abcde");
            set_a(-1, "abcdefgh");
            r = read_label(get_a, 512);
            rb = read_label(get_b, 512);
            check.expect(label_is(r, "abcdefgh") && label_is(rb, "abcde"),
                         N("two objects with different labels keep them apart; labels need not be unique (§18.7)"), describe(r) + " | " + describe(rb));
            set_b(-1, "abcdefgh");
            check.expect(label_is(read_label(get_b, 512), "abcdefgh"), N("the same label can be given to two objects"));

            set_a(-1, "abcdef");
            // bufSize truncation (section 18.9): at most bufSize - 1 characters plus the terminator are written.
            r = read_label(get_a, 4);
            check.expect(r.text == "abc" && r.length == 3 && r.terminated && r.tail_intact,
                         N("GetObjectLabel with bufSize 4 on \"abcdef\": \"abc\" + terminator, length 3, nothing written beyond"), describe(r));
            r = read_label(get_a, 6);
            check.expect(r.text == "abcde" && r.length == 5 && r.terminated && r.tail_intact,
                         N("bufSize 6: \"abcde\" + terminator, length 5 (one character cut)"), describe(r));
            r = read_label(get_a, 7);
            check.expect(label_is(r, "abcdef"), N("bufSize 7 (= length + 1) returns the complete label"), describe(r));
            r = read_label(get_a, 1);
            check.expect(r.text.empty() && r.length == 0 && r.terminated && r.tail_intact,
                         N("bufSize 1: only the terminator is written, length 0"), describe(r));

            if (std::string_view(ns) == "buffer")
            {
                // bufSize 0 cannot even hold the terminator; section 18.9 does not say what length is returned, so the
                // outcome is only printed.
                std::array<char, 8> none{};
                none.fill('#');
                GLsizei zero_len = -1;
                get_a(0, &zero_len, none.data());
                std::cout << std::format("   note: GetObjectLabel with bufSize 0 returned length {}, buffer {} (spec silent; not asserted)\n", zero_len,
                                         none[0] == '#' ? "untouched" : "written");
                (void)demo::drain_gl_errors();
            }

            GLsizei length = -1;
            get_a(100, &length, nullptr);
            check.expect_eq(length, 6, N("GetObjectLabel with label = NULL returns only the length (6) (§18.9)"));
            std::array<char, 16> buf{};
            buf.fill('#');
            get_a(16, nullptr, buf.data());
            check.expect(std::string_view(buf.data()) == "abcdef", N("GetObjectLabel with length = NULL still returns the string"));
            check.expect_no_gl_error(N("the NULL-length and NULL-label queries raise no error"));

            set_a(-1, nullptr);
            r = read_label(get_a, 512);
            check.expect(label_is(r, ""), N("ObjectLabel with label = NULL removes the label (§18.7)"), describe(r));

            // MAX_LABEL_LENGTH (section 18.7): the label must be strictly shorter.
            set_a(-1, "before");
            const std::string too_long(static_cast<std::size_t>(max), 'L');
            set_a(-1, too_long.c_str());
            check.expect_gl_error(EC::InvalidValue, N("a null-terminated label of MAX_LABEL_LENGTH characters -> INVALID_VALUE"));
            set_a(max, too_long.data());
            check.expect_gl_error(EC::InvalidValue, N("a label with length = MAX_LABEL_LENGTH -> INVALID_VALUE"));
            r = read_label(get_a, 512);
            check.expect_or_known_deviation(label_is(r, "before"), N("the failed calls left the previous label in place (Table 2.3: INVALID_VALUE ignores the command)"),
                                            label_is(r, "before") ? std::string{} : std::format("label now has length {}", r.length), "Mesa",
                                            "Mesa raises INVALID_VALUE for a label of MAX_LABEL_LENGTH characters but still stores it");
            const std::string fits(static_cast<std::size_t>(max - 1), 'M');
            set_a(-1, fits.c_str());
            check.expect_no_gl_error(N("a label of MAX_LABEL_LENGTH - 1 characters is accepted"));
            r = read_label(get_a, max);
            check.expect(label_is(r, fits), N("...and read back completely with bufSize = MAX_LABEL_LENGTH (length MAX_LABEL_LENGTH - 1)"), std::format("length={}", r.length));

            // A negative bufSize (section 18.9).
            std::array<char, 8> small{};
            small.fill('#');
            GLsizei len2 = -1;
            get_a(-1, &len2, small.data());
            check.expect_gl_error(EC::InvalidValue, N("GetObjectLabel with a negative bufSize -> INVALID_VALUE"));
            check.expect(len2 == -1 && small[0] == '#', N("...and no value is written through the pointer arguments (§2.3.1)"),
                         std::format("length={} first byte '{}'", len2, small[0]));

            set_a(-1, "x");
        }

        /// Both entry points reject an invalid name with INVALID_VALUE.
        void expect_label_invalid(demo::Check& check, const std::string& what, const SetLabel& set, const GetLabel& get)
        {
            set(-1, "label");
            check.expect_gl_error(EC::InvalidValue, std::format("{}: ObjectLabel -> INVALID_VALUE", what));
            std::array<char, 16> buf{};
            buf.fill('#');
            GLsizei len = -1;
            get(16, &len, buf.data());
            check.expect_gl_error(EC::InvalidValue, std::format("{}: GetObjectLabel -> INVALID_VALUE", what));
            check.expect(len == -1 && buf[0] == '#', std::format("{}: ...and writes neither length nor label (§2.3.1)", what),
                         std::format("length={} first byte '{}'", len, buf[0]));
        }

        // Negative controls (the checks above can fail).
        void area_negative(const demo::Env& /*env*/, demo::Check& check)
        {
            // 1. A sink-based check observes the absence of a message: with the callback removed, the same insert
            //    that was delivered a moment ago is not seen by the sink (it goes to the log instead).
            insert_z(Src::Application, Typ::Marker, 9, Sev::High, "probe");
            const std::size_t before = sink_.size();
            (void)sink_.take();
            mgl::glDebugMessageCallback(nullptr, nullptr);
            insert_z(Src::Application, Typ::Marker, 9, Sev::High, "probe");
            const std::size_t after = sink_.size();
            check.expect(before == 1 && after == 0 && geti(mgl::GetParameter::DebugLoggedMessages) == 1,
                         "negative control: without a callback the sink sees nothing (the delivery checks would FAIL) and the message is in the log",
                         std::format("before={} after={}", before, after));
            drain_log();
            mgl::glDebugMessageCallback(&debug_callback_a, &sink_);

            // 2. A filter check is sensitive to the filter: the same message that was delivered under the baseline is
            //    blocked after a one-call mutation of the state, so "delivered" and "blocked" expectations differ.
            const bool on = delivered(Src::Application, Typ::Marker, 77, Sev::High);
            mgl::glDebugMessageControl(Src::DontCare, Typ::DontCare, Sev::High, 0, nullptr, GL_FALSE);
            const bool off = delivered(Src::Application, Typ::Marker, 77, Sev::High);
            check.expect(on && !off, "negative control: one Control(HIGH, FALSE) call flips the outcome of the same insert (delivered -> blocked)",
                         std::format("on={} off={}", on, off));

            // 3. A label check is sensitive to the label.
            const GLuint name = vbo_.value;
            mgl::glObjectLabel(Ns::Buffer, name, -1, "changed");
            const LabelRead changed = read_label([&](GLsizei b, GLsizei* l, GLchar* s) { mgl::glGetObjectLabel(Ns::Buffer, name, b, l, s); }, 64);
            check.expect(!label_is(changed, kSceneLabels[2]) && label_is(changed, "changed"),
                         "negative control: relabelling the vertex buffer makes the comparison with the original label fail", describe(changed));
            mgl::glObjectLabel(Ns::Buffer, name, -1, kSceneLabels[2]);
        }

        // ---------------------------------------------------------------------------------------------------------
        // Label subjects: one real object per namespace of Table 18.4
        // ---------------------------------------------------------------------------------------------------------

        static const std::vector<LabelSubject>& label_subjects()
        {
            static const std::vector<LabelSubject> subjects = {
                {"buffer", Ns::Buffer,
                 [] {
                     mgl::BufferId id{};
                     mgl::glGenBuffers(1, &id);
                     mgl::glBindBuffer(mgl::BufferTarget::Array, id);  // a name becomes a buffer object when first bound
                     mgl::glBindBuffer(mgl::BufferTarget::Array, mgl::BufferId{0});
                     return id.value;
                 },
                 [](GLuint name) {
                     const mgl::BufferId id{name};
                     mgl::glDeleteBuffers(1, &id);
                 }},
                {"shader", Ns::Shader, [] { return mgl::glCreateShader(mgl::ShaderType::Fragment).value; },
                 [](GLuint name) { mgl::glDeleteShader(mgl::ShaderId{name}); }},
                {"program", Ns::Program, [] { return mgl::glCreateProgram().value; },
                 [](GLuint name) { mgl::glDeleteProgram(mgl::ProgramId{name}); }},
                {"vertex array", Ns::VertexArray,
                 [] {
                     mgl::VertexArrayId id{};
                     mgl::glGenVertexArrays(1, &id);
                     mgl::glBindVertexArray(id);
                     mgl::glBindVertexArray(mgl::VertexArrayId{0});
                     return id.value;
                 },
                 [](GLuint name) {
                     const mgl::VertexArrayId id{name};
                     mgl::glDeleteVertexArrays(1, &id);
                 }},
                {"query", Ns::Query,
                 [] {
                     mgl::QueryId id{};
                     mgl::glGenQueries(1, &id);
                     mgl::glBeginQuery(mgl::QueryTarget::AnySamplesPassed, id);  // a query object exists after its first Begin
                     mgl::glEndQuery(mgl::QueryTarget::AnySamplesPassed);
                     return id.value;
                 },
                 [](GLuint name) {
                     const mgl::QueryId id{name};
                     mgl::glDeleteQueries(1, &id);
                 }},
                {"program pipeline", Ns::ProgramPipeline,
                 [] {
                     mgl::ProgramPipelineId id{};
                     mgl::glGenProgramPipelines(1, &id);
                     mgl::glBindProgramPipeline(id);
                     mgl::glBindProgramPipeline(mgl::ProgramPipelineId{0});
                     return id.value;
                 },
                 [](GLuint name) {
                     const mgl::ProgramPipelineId id{name};
                     mgl::glDeleteProgramPipelines(1, &id);
                 }},
                {"sampler", Ns::Sampler,
                 [] {
                     mgl::SamplerId id{};
                     mgl::glGenSamplers(1, &id);
                     mgl::glBindSampler(0, id);
                     mgl::glBindSampler(0, mgl::SamplerId{0});
                     return id.value;
                 },
                 [](GLuint name) {
                     const mgl::SamplerId id{name};
                     mgl::glDeleteSamplers(1, &id);
                 }},
                {"texture", Ns::Texture,
                 [] {
                     mgl::TextureId id{};
                     mgl::glGenTextures(1, &id);
                     mgl::glBindTexture(mgl::TextureTarget::Texture2D, id);
                     mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
                     return id.value;
                 },
                 [](GLuint name) {
                     const mgl::TextureId id{name};
                     mgl::glDeleteTextures(1, &id);
                 }},
                {"renderbuffer", Ns::Renderbuffer,
                 [] {
                     mgl::RenderbufferId id{};
                     mgl::glGenRenderbuffers(1, &id);
                     mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, id);
                     mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
                     return id.value;
                 },
                 [](GLuint name) {
                     const mgl::RenderbufferId id{name};
                     mgl::glDeleteRenderbuffers(1, &id);
                 }},
                {"framebuffer", Ns::Framebuffer,
                 [] {
                     mgl::FramebufferId id{};
                     mgl::glGenFramebuffers(1, &id);
                     mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, id);
                     mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
                     return id.value;
                 },
                 [](GLuint name) {
                     const mgl::FramebufferId id{name};
                     mgl::glDeleteFramebuffers(1, &id);
                 }},
                {"transform feedback", Ns::TransformFeedback,
                 [] {
                     mgl::TransformFeedbackId id{};
                     mgl::glGenTransformFeedbacks(1, &id);
                     mgl::glBindTransformFeedback(mgl::TransformFeedbackTarget::TransformFeedback, id);
                     mgl::glBindTransformFeedback(mgl::TransformFeedbackTarget::TransformFeedback, mgl::TransformFeedbackId{0});
                     return id.value;
                 },
                 [](GLuint name) {
                     const mgl::TransformFeedbackId id{name};
                     mgl::glDeleteTransformFeedbacks(1, &id);
                 }},
            };
            return subjects;
        }

        // ---------------------------------------------------------------------------------------------------------
        // Panel
        // ---------------------------------------------------------------------------------------------------------

        void fill(int x, int y, int w, int h, float r, float g, float b)
        {
            mgl::glScissor(x, y, w, h);
            mgl::glClearColor(r, g, b, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
        }

        void draw_panel(const demo::Env& env)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glEnable(mgl::Capability::ScissorTest);
            const int total = kPanelColumns * kPanelRows;
            for (int i = 0; i < total; ++i)
            {
                const int col = i % kPanelColumns;
                const int row = i / kPanelColumns;
                const int x = col * kCell;
                const int y = row * kCell;
                // Cell rim: dark, so adjacent cells stay distinguishable.
                fill(x, y, kCell, kCell, 0.05f, 0.05f, 0.06f);
                if (static_cast<std::size_t>(i) >= areas_.size()) continue;
                const AreaResult& a = areas_[static_cast<std::size_t>(i)];
                const bool pass = a.failed == 0 && a.checks > 0;
                if (pass) fill(x + 2, y + 2, kCell - 4, kCell - 4, 0.10f, 0.72f, 0.20f);
                else fill(x + 2, y + 2, kCell - 4, kCell - 4, 0.85f, 0.10f, 0.10f);
            }
            // Strip: one square per scene object label.
            const int strip_y = kPanelRows * kCell;
            for (std::size_t i = 0; i < scene_label_ok_.size(); ++i)
            {
                const int x = static_cast<int>(i) * kCell;
                fill(x, strip_y, kCell, kStripHeight, 0.05f, 0.05f, 0.06f);
                if (scene_label_ok_[i]) fill(x + 2, strip_y + 2, kCell - 4, kStripHeight - 4, 0.10f, 0.72f, 0.20f);
                else fill(x + 2, strip_y + 2, kCell - 4, kStripHeight - 4, 0.85f, 0.10f, 0.10f);
            }
            mgl::glDisable(mgl::Capability::ScissorTest);
            (void)env;
        }

        void print_summary() const
        {
            std::cout << "\n   areas:\n";
            for (const AreaResult& a : areas_) std::cout << std::format("     {:<34} {:>3} check(s), {} failed\n", a.name, a.checks, a.failed);
        }

        // ---------------------------------------------------------------------------------------------------------
        // State
        // ---------------------------------------------------------------------------------------------------------

        /// The debug state as the context was created, recorded in init() before anything changed it.
        struct Initial
        {
            GLint flags = 0;
            bool debug_output = false;
            bool synchronous = true;
            GLDEBUGPROC callback = nullptr;
            void* user = nullptr;
            GLint depth = 0;
            GLint logged = -1;
            GLint next_length = -1;
            GLint max_message_length = 0;
            GLint max_logged = 0;
            GLint max_depth = 0;
            GLint max_label = 0;
            GLint log_count_after_inserts = -1;
            std::vector<GLuint> log_ids;
            bool log_sources_application = false;
            std::vector<GLuint> callback_ids;
        };

        Sink sink_;
        Sink sink_b_;
        Initial initial_;
        std::vector<Message> scene_stream_;
        std::vector<AreaResult> areas_;
        std::array<bool, 5> scene_label_ok_{};

        int scene_y_ = 0;
        int scene_h_ = 0;

        mgl::ProgramId program_{};
        mgl::VertexArrayId vao_{};
        mgl::BufferId vbo_{};
        mgl::TextureId texture_{};
        mgl::SamplerId sampler_{};
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es32-debug-output",
        .title = "Debug output: messages, groups and labels",
        .min_es = {3, 2},
        .summary = "OpenGL ES 3.2 debug output in a debug context: glDebugMessageCallback + glGetPointerv, DEBUG_OUTPUT and DEBUG_OUTPUT_SYNCHRONOUS, "
                   "glDebugMessageInsert, glDebugMessageControl filtering, the message log, debug groups (glPushDebugGroup/glPopDebugGroup) "
                   "and object labels for all 11 namespaces plus sync objects.",
        .source_dir = DEMO_SOURCE_DIR,
        .debug_context = true,
    };
    return demo::run<DebugOutput>(argc, argv, info);
}
