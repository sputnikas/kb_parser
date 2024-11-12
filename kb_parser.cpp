#define __STDC_LIB_EXT1__
#include <bit>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <regex>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/quaternion.hpp>
namespace fs = std::filesystem;

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "tiny_gltf.h"

using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Вспомогательные функции
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
T max_array(T* a, int init, int offset, int num_a) {
    T result = a[init];
    init += offset;
    while (init < num_a) {
        if (result < a[init]) {
            result = a[init];
        }
        init += offset;
    }
    return result;
}

template <typename T>
T min_array(T* a, int init, int offset, int num_a) {
    T result = a[init];
    init += offset;
    while (init < num_a) {
        if (result > a[init]) {
            result = a[init];
        }
        init += offset;
    }
    return result;
}

template <typename T>
std::string toString(T val) {
    std::ostringstream oss;
    oss << val;
    return oss.str();
}

void print_uarray(unsigned char* arr, int num) {
    printf("{ ");
    int i = 0;
    while (i < num - 1) {
        printf("%d, ", arr[i]);
        ++i;
    }
    printf("%d }\n", arr[num - 1]);
}

void print_farray(float* arr, int num) {
    printf("{ ");
    int i = 0;
    while (i < num - 1) {
        printf("%.15e, ", arr[i]);
        ++ i;
    }
    printf("%.15e }\n", arr[num - 1]);
}

namespace glm {
bool operator<(const glm::vec3 a, const glm::vec3 b) {
    if (a.x < b.x) {
        return 1;
    } else if (a.x == b.x) {
        if (a.y < b.y) {
            return 1;
        } else if (a.y == b.y) {
            return a.z < b.z;
        }
    }
    return 0;
}
}  // namespace glm

float max_abs(float a1, float a2, float a3) {
    if ((fabs(a1) > fabs(a2)) && (fabs(a1) > fabs(a3))) {
        return a1;
    } else {
        if (fabs(a2) > fabs(a3)){
            return a2;
        } else {
            return a3;
        }
    }
}

void calc_quaternion(const glm::mat4 &ibmg, float &x, float &y, float &z, float &a, float &px, float &py, float &pz) {
    glm::mat4 bmg = glm::inverse(ibmg);
    glm::mat4 T = glm::mat4(1.0);
    T[3][0] = bmg[3][0];
    T[3][1] = bmg[3][1];
    T[3][2] = bmg[3][2];
    px = ibmg[3][0];
    py = ibmg[3][1];
    pz = ibmg[3][2];
    glm::mat4 R = glm::transpose(glm::inverse(T) * bmg); // Оказалось, что матрица в glm хранится по столбцам! То есть транспонированы!
    x = sqrtf(fabs(1 + R[0][0] - R[1][1] - R[2][2])) / 2;
    y = sqrtf(fabs(1 + R[1][1] - R[0][0] - R[2][2])) / 2;
    z = sqrtf(fabs(1 + R[2][2] - R[1][1] - R[0][0])) / 2;
    float xy = (R[1][0] + R[0][1]) / 4;
    float xz = (R[2][0] + R[0][2]) / 4;
    float yz = (R[1][2] + R[2][1]) / 4;
    float ax = -(R[2][1] - R[1][2]) / 4;
    float ay = (R[2][0] - R[0][2]) / 4;
    float az = -(R[1][0] - R[0][1]) / 4;
    if (xy < 0 && xz < 0) {
        x = -x;
    }
    if (xy < 0 && yz < 0) {
        y = -y;
    }
    if (xz < 0 && yz < 0) {
        z = -z;
    }
    a = 1.0f;
    if (x * x + y * y + z * z > 0.00001f) {
        a = max_abs(ax, ay, az) / max_abs(x, y, z);
    }
    float length = sqrtf(x * x + y * y + z * z + a * a);
    x /= length;
    y /= length;
    z /= length;
    a /= length;
}

char* mem_find(void *a, long lena, const char *b, long lenb) {
    char *pointer = (char *) a;
    int i = 0;
    while (i < lena - lenb) {
        int rc = std::memcmp(pointer, b, lenb);
        if (rc == 0) {
            return pointer;
        }
        ++pointer;
        ++i;
    }
    return NULL;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// BSAParser - анимации хранятся в отдельных файлах .bsa
///////////////////////////////////////////////////////////////////////////////////////////////////

struct BSAFrame {
    unsigned int frame_check;
    float* joints;
};

class BSAParser {
   public:
    BSAParser(const string& filename);
    ~BSAParser();
    ///////////////////////////////////////////////////////////////////////////////////////////////
    void toGLTF(tinygltf::Model& m, const string& anim_name);
    ///////////////////////////////////////////////////////////////////////////////////////////////
    unsigned short a1;
    unsigned short a2;
    unsigned int header_check;
    unsigned int num_frames;
    unsigned int num_joints;
    unsigned int fps;
    float* joints;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// BSAParser implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

BSAParser::BSAParser(const string& filename) {
    cout << filename << endl;
#ifdef __STDC_LIB_EXT1__
    FILE* f;
    fopen_s(&f, filename.c_str(), "rb");
#else
    FILE* f = fopen(filename.c_str(), "rb");
#endif
    fread(&a1, sizeof(a1), 1, f);
    fread(&a2, sizeof(a2), 1, f);
    fread(&header_check, sizeof(header_check), 1, f);
    fread(&num_frames, sizeof(num_frames), 1, f);
    fread(&num_joints, sizeof(num_joints), 1, f);
    fread(&fps, sizeof(fps), 1, f);
    BSAFrame* frames = new BSAFrame[num_frames];
    for (unsigned i = 0; i < num_frames; ++i) {
        frames[i].joints = new float[7 * num_joints];
        for (unsigned j = 0; j < 7 * num_joints; ++j) {
            frames[i].joints[j] = 0.0f;
            if (j % 4 == 3) {
                frames[i].joints[j] = 1.0f;
            }
        }
        fread(&frames[i].frame_check, sizeof(unsigned int), 1, f);
        fread(&(frames[i].joints[0]), sizeof(float), 7 * num_joints, f);
    }
    if (fps == 0) {
        fps = 30;
    }
    joints = new float[7 * num_joints * num_frames];  // сначала translations затем rotations
    for (unsigned i = 0; i < num_joints; ++i) {
        for (unsigned j = 0; j < num_frames; ++j) {
            // translations
            joints[3 * i * num_frames + 3 * j] = frames[j].joints[7 * i + 4];
            joints[3 * i * num_frames + 3 * j + 1] = frames[j].joints[7 * i + 5];
            joints[3 * i * num_frames + 3 * j + 2] = frames[j].joints[7 * i + 6];
            float quat2 = powf(frames[j].joints[7 * i], 2) + powf(frames[j].joints[7 * i + 1], 2) + powf(frames[j].joints[7 * i + 2], 2) + powf(frames[j].joints[7 * i + 3], 2);
            float quat = sqrtf(quat2);
            joints[3 * num_joints * num_frames + 4 * i * num_frames + 4 * j] = frames[j].joints[7 * i] / quat;
            joints[3 * num_joints * num_frames + 4 * i * num_frames + 4 * j + 1] = frames[j].joints[7 * i + 1] / quat;
            joints[3 * num_joints * num_frames + 4 * i * num_frames + 4 * j + 2] = frames[j].joints[7 * i + 2] / quat;
            joints[3 * num_joints * num_frames + 4 * i * num_frames + 4 * j + 3] = frames[j].joints[7 * i + 3] / quat;
        }
    }
    for (unsigned i = 0; i < num_frames; ++i) {
        delete[] frames[i].joints;
    }
    delete[] frames;
    fclose(f);
}

BSAParser::~BSAParser() { 
    delete[] joints;
}

void BSAParser::toGLTF(tinygltf::Model& m, const string& anim_name) {
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Буферы анимации
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Анимация bsa представляет собой набор key-фреймов
    ///////////////////////////////////////////////////////////////////////////////////////////////

    unsigned int buffer_index = (unsigned int) m.buffers.size();
    unsigned int buffer_view_index = (unsigned int) m.bufferViews.size();
    unsigned int accessor_index = (unsigned int) m.accessors.size();
    // Времена кадров
    float* times = new float[num_frames];
    for (unsigned int i = 0; i < num_frames; ++i) {
        times[i] = 1.0f * i / fps;
    }

    tinygltf::Buffer b_times;
    tinygltf::BufferView bv_times;
    tinygltf::Accessor a_times;

    b_times.data.resize(num_frames * sizeof(float));
    memcpy(&b_times.data[0], &times[0], num_frames * sizeof(float));
    float tmin = min_array<float>(times, 0, 1, num_frames);
    float tmax = max_array<float>(times, 0, 1, num_frames);
    delete [] times;

    bv_times.buffer = buffer_index;
    bv_times.byteOffset = 0;
    bv_times.byteLength = num_frames * sizeof(float);

    a_times.bufferView = buffer_view_index;
    a_times.byteOffset = 0;
    a_times.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    a_times.count = num_frames;
    a_times.type = TINYGLTF_TYPE_SCALAR;
    a_times.maxValues = {tmax};
    a_times.minValues = {tmin};

    m.buffers.push_back(b_times);
    m.bufferViews.push_back(bv_times);
    m.accessors.push_back(a_times);

    // Буферы кватернионов и трансляций
    tinygltf::Buffer b_anims;
    vector<tinygltf::BufferView> bv_anims;
    vector<tinygltf::Accessor> a_anims_rotations;
    vector<tinygltf::Accessor> a_anims_translations;

    // buffer будет общий (Общий нельзя! Ошибка - bufferView.byteStride must not be defined for buffer views used by animation sampler accessors.)
    b_anims.data.resize(7 * num_joints * num_frames * sizeof(float));
    memcpy(&b_anims.data[0], &joints[0], 7 * num_joints * num_frames * sizeof(float));
    m.buffers.push_back(b_anims);
    // Проходимся по костям и создаём для каждой кости bufferView и accessor отдельно для
    // трансляций, отдельно для кватернионов
    tinygltf::Animation anim;
    anim.name = anim_name;

    unsigned index_buffer = (unsigned)m.buffers.size() - 1;
    unsigned index_times = (unsigned)m.accessors.size() - 1;
    for (unsigned i = 0; i < num_joints; ++i) {
        unsigned index_buffer_view = (unsigned)m.bufferViews.size();
        unsigned index_accessor = (unsigned)m.accessors.size();
        unsigned index_sampler = (unsigned)anim.samplers.size();

        tinygltf::BufferView bv_anims_rotation;
        tinygltf::BufferView bv_anims_translation;

        bv_anims_translation.buffer = index_buffer;
        bv_anims_translation.byteOffset = 3 * i * num_frames * sizeof(float);
        bv_anims_translation.byteLength = 3 * num_frames * sizeof(float);

        bv_anims_rotation.buffer = index_buffer;
        bv_anims_rotation.byteOffset = 3 * num_joints * num_frames * sizeof(float) + 4 * i * num_frames * sizeof(float);
        bv_anims_rotation.byteLength = 4 * num_frames * sizeof(float);

        m.bufferViews.push_back(bv_anims_translation);
        m.bufferViews.push_back(bv_anims_rotation);

        tinygltf::Accessor a_anims_translation;
        tinygltf::Accessor a_anims_rotation;

        a_anims_translation.bufferView = index_buffer_view;
        a_anims_translation.byteOffset = 0;
        a_anims_translation.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        a_anims_translation.count = num_frames;
        a_anims_translation.type = TINYGLTF_TYPE_VEC3;

        a_anims_rotation.bufferView = index_buffer_view + 1;
        a_anims_rotation.byteOffset = 0;
        a_anims_rotation.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        a_anims_rotation.count = num_frames;
        a_anims_rotation.type = TINYGLTF_TYPE_VEC4;

        m.accessors.push_back(a_anims_translation);
        m.accessors.push_back(a_anims_rotation);

        // Готовим анимации - samler - безье кривая, от времени
        tinygltf::AnimationSampler anim_smp1;
        anim_smp1.interpolation = "LINEAR";
        anim_smp1.input = index_times;      // accessor to times
        anim_smp1.output = index_accessor;  // accessor to translations

        tinygltf::AnimationSampler anim_smp2;
        anim_smp2.interpolation = "LINEAR";
        anim_smp2.input = index_times;          // accessor to times
        anim_smp2.output = index_accessor + 1;  // accessor to rotations

        anim.samplers.push_back(anim_smp1);
        anim.samplers.push_back(anim_smp2);

        tinygltf::AnimationChannel anim_chan1;
        anim_chan1.sampler = index_sampler;
        anim_chan1.target_node = i;
        anim_chan1.target_path = "translation";

        tinygltf::AnimationChannel anim_chan2;
        anim_chan2.sampler = index_sampler + 1;
        anim_chan2.target_node = i;
        anim_chan2.target_path = "rotation";

        anim.channels.push_back(anim_chan1);
        anim.channels.push_back(anim_chan2);
    }

    // printf("target_node = %d\n", anim.channels[0].target_node);
    m.animations.push_back(anim);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Структура файлов King's Bounty сильно отличается от структуры файлов Эадора
// Поэтому здесь предстоит непростой путь к пониманию, что за что отвечает
///////////////////////////////////////////////////////////////////////////////////////////////////

struct Weight {
    unsigned int joint;
    float weight;
};

struct Weights {
    vector<Weight> weights;
};

struct WeightsMap {
    map<unsigned int, float> weights;
};

struct Joint {
    int size_name;
    char name[64];
    int parent;
    float position[10];
    int unknown;
    float matrix[16];
};

struct BMSGroup {
    long V_format;
    long material_id;
    long indeks_offset;
    long indeks_size;
    long vertex_offset;
};

struct BMAGroup {
    long V_format;
    long material_id;
    unsigned int indeks_offset;
    unsigned int indeks_size;
    long vertex_offset;
    long bones_per_vertex;
    unsigned char bone_remap[16];
};

struct BMAFileFormat3 {
    short bm_x;
    short bm_y;
    short bm_z;
    short bm_scale;  // главная штука для расшифровки вершин?
    long bm_vector_n;
    short bm_u;
    short bm_v;
    unsigned char bones[4];
    unsigned char weights[4];
};

struct BMAFileFormat7 {
    short bm_x;
    short bm_y;
    short bm_z;
    short bm_scale;  // главная штука для расшифровки вершин?
    long bm_vector_n;
    long bm_vector_s;
    long bm_vector_t;
    short bm_u;
    short bm_v;
    unsigned char bones[4];
    unsigned char weights[4];
};

struct VertexAdditional {
    unsigned int bones_remap[4];
    map<int, int> parts;
    //
    ~VertexAdditional();
};

VertexAdditional::~VertexAdditional() {
    parts.clear();
}

struct Vertex {
    float x;
    float y;
    float z;
    float nx;
    float ny;
    float nz;
    float u;
    float v;
    unsigned char bones[4];
    float weights[4];
};

class KBParser {
   public:
    KBParser(const string& filename, const string& folder);
    ~KBParser();
    // Функции
    void toGLTF(const string& filename, const string& bsa_path, const string& texture_path, const string& out_path, const string& prefix);

   private:
    unsigned int version;
    unsigned int check;
    unsigned int scaled;
    unsigned int mainscale;
    unsigned int bma_or_bms;  // joints_check!
    unsigned int n_joints;
    unsigned int material_check;
    unsigned int n_materials;
    unsigned int parts_check;
    unsigned int n_parts;
    unsigned int vertices_check;
    unsigned int n_vertices;
    unsigned int indices_check;
    unsigned int n_indices;
    // Arrays
    Vertex minVertex;
    Vertex maxVertex;
    Vertex* vertices;
    unsigned short* indices;
    Joint* joints;
    vector<string> texDiffuse;
    vector<string> texSpecular;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// KBParser implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

KBParser::KBParser(const string& filename, const string& folder) {
    string filename2 = folder + filename;
#ifdef __STDC_LIB_EXT1__
    FILE* f;
    fopen_s(&f, filename2.c_str(), "rb");
#else
    FILE* f = fopen(filename.c_str(), "rb");
#endif
    // Всё начинается с 6 неизвестных int
    // int unknown[100];
    fread(&version, sizeof(int), 1, f);  // Версия файла
    printf("Version = %d\n", version);
    fread(&check, sizeof(int), 1, f);  // BMA/BMS метка = 0x0000A34B
    printf("Check = %08X = 0x0000A34B\n", check);
    fread(&scaled, sizeof(int), 1, f);
    printf("Scaled = %d\n", scaled);
    if (scaled & 0x20000000) {
        fread(&mainscale, sizeof(int), 1, f);
        printf("Mainscale = %d\n", mainscale);
    }
    fread(&bma_or_bms, sizeof(int), 1, f);
    printf("BMA_or_BMS = %X = 0x00001441/0x0005535B\n", bma_or_bms);
    printf("ftell(f) = %x\n", ftell(f));
    if (bma_or_bms == 0x00001441) {
        fread(&n_joints, sizeof(int), 1, f);
        printf("Num_joints = %d\n", n_joints);
        int c = 1;
        int j = 0;
        joints = new Joint[n_joints];
        for (unsigned i = 0; i < n_joints; ++i) {
            fread(&joints[i].size_name, sizeof(int), 1, f);
            fread(&joints[i].name, sizeof(char), joints[i].size_name + 1, f);
            fread(&joints[i].parent, sizeof(int), 1, f);
            fread(&joints[i].position[0], sizeof(float), 10, f);
            fread(&joints[i].unknown, sizeof(int), 1, f);
            fread(&joints[i].matrix[0], sizeof(float), 16, f);
            printf("    Joint[%2d].Name = %20s ", i, joints[i].name);
            printf(" .Parent = %2d ", joints[i].parent);
            printf(" .Unknown = %2d", joints[i].unknown);
            printf(" .Position[9] = %2.8f\n", joints[i].position[9]);
            printf("    ");
            print_farray(&joints[i].position[0], 10);
            printf("    ");
            print_farray(&joints[i].matrix[0], 16);
        }
    }
    printf("ftell(f) = %x\n", ftell(f));
    fread(&material_check, sizeof(int), 1, f);
    printf("Material_check = %08X\n", material_check);
    fread(&n_materials, sizeof(int), 1, f);
    printf("N_materials = %d\n", n_materials);
    // Материалы в KB это полная жесть, параметров жуть, есть деревья параметров!
    // Что за что отвечает неизвестно!
    long size_material_block;
    char* material_block = new char[1024 * n_materials];
    long begin_material_block = ftell(f);
    unsigned int end_material_block = 0;
    long in_material_index = 0;
    while (end_material_block != 0x02238ECB) {
        fseek(f, begin_material_block + in_material_index, SEEK_SET);
        fread(&end_material_block, sizeof(int), 1, f);
        in_material_index++;
    }
    long end_material = ftell(f) - 4;
    size_material_block = end_material - begin_material_block;
    /*
    fseek(f, begin_material_block, SEEK_SET);
    map<string, int> mat_sparams;
    map<string, string> mat_params;
    mat_sparams["k_specularenvspecmask"] = 61; // strange value
    mat_sparams["diffuse"] = 61;  // strange value
    mat_sparams["texDiffuse"] = -1; // string value
    mat_sparams["texSpecular"] = -1;
    mat_sparams["texEnv"] = -1;
    mat_sparams["cLuminosity"] = -2; // vec4float value
    mat_sparams["cDiffuse"] = -2;
    mat_sparams["cSpecular"] = -2;
    mat_sparams["cEnv"] = -2;
    mat_sparams["fShadowTransp"] = -3; // float value
    mat_sparams["tgDiffuse"] = -4; // int[4] + float[16] value
    mat_sparams["tgSpecular"] = -4;
    char buf[512];
    mat_params["texDiffuse"] = filename.substr(filename.size() - 4) + "_diff.dds";
    mat_params["texSpecular"] = filename.substr(filename.size() - 4) + "_spec.dds";
    int var_size;
    while (ftell(f) != end_material) {
        fread(&var_size, sizeof(int), 1, f);
        fread(&buf[0], sizeof(char), var_size + 1, f);
        string var = string(buf);
        if (mat_sparams.find(var) != mat_sparams.end()) {
            if (mat_sparams[var] > 0) {
                fread(&buf[0], sizeof(char), mat_sparams[var], f);
                buf[mat_sparams[var] + 1] = '\0';
            } else {
                int type_param;
                fread(&type_param, sizeof(int), 1, f);
                switch (type_param) {
                    case 6:
                        int value_size;
                        fread(&value_size, sizeof(int), 1, f);
                        fread(&buf[0], sizeof(char), value_size + 1, f);
                        break;
                    case 4:
                        float vecf[4];
                        fread(&vecf[0], sizeof(float), 4, f);
#ifdef __STDC_LIB_EXT1__
                        sprintf_s(&buf[0], sizeof(buf), "{%f, %f, %f, %f}", vecf[0], vecf[1], vecf[2], vecf[3]);
#else
                        sprintf(&buf[0], "{%f, %f, %f, %f}", vecf[0], vecf[1], vecf[2], vecf[3]);
#endif
                        break;
                    case 2:
                        float a2;
                        fread(&a2, sizeof(float), 1, f);
#ifdef __STDC_LIB_EXT1__
                        sprintf_s(&buf[0], sizeof(buf), "%f", a2);
#else
                        sprintf(&buf[0], "%f", a2);
#endif
                        break;
                    case 5:
                        int b[4]; // mask
                        float a5[16]; // matrix 4x4
                        fread(&b[0], sizeof(int), 4, f);
                        fread(&a5[0], sizeof(float), 16, f);
#ifdef __STDC_LIB_EXT1__
                        sprintf_s(&buf[0], sizeof(buf), "mask {%x, %x, %x, %x} mat4 {{%f %f %f %f}, {%f %f %f %f}, {%f, %f, %f, %f}, {%f, %f, %f, %f}}", 
                            b[0], b[1], b[2], b[3], 
                            a5[0], a5[1], a5[2], a5[3], a5[4], a5[5], a5[6], a5[7], a5[8], a5[9], a5[10], a5[11], a5[12], a5[13], a5[14], a5[15]
                        );
#else
                        sprintf(&buf[0], "mask {%x, %x, %x, %x} mat4 {{%f %f %f %f}, {%f %f %f %f}, {%f, %f, %f, %f}, {%f, %f, %f, %f}}", 
                            b[0], b[1], b[2], b[3], 
                            a5[0], a5[1], a5[2], a5[3], a5[4], a5[5], a5[6], a5[7], a5[8], a5[9], a5[10], a5[11], a5[12], a5[13], a5[14], a5[15]
                        );
#endif
                        break;
                }
            }
        }
        mat_params[var] = string(buf);
        cout << var << " = " << mat_params[var] << endl;
    } // Неудачная идея распознать все параметры материалов - работает не для всех моделей, особенно большие вопросы к diffuse материалам - у них разные header'ы*/
    fseek(f, begin_material_block, SEEK_SET);
    fread(material_block, sizeof(char), size_material_block, f);  // Загоняем блок материлов пока в отдельный буфер
    // Так как я всё равно шейдеры как в kb не напишу, ищем в параметрах материала texDiffuse и texSpecular
    char buf[256];
    for (unsigned int i = 0; i < n_materials; i++) {
        texDiffuse.push_back(filename.substr(filename.size() - 4) + "_diff.dds");
        texSpecular.push_back(filename.substr(filename.size() - 4) + "_spec.dds");
    }
    char* pointer = material_block;
    int mat_index = 0;
    while (pointer != NULL) {
        pointer = mem_find(pointer, size_material_block - (long) (pointer - material_block), "texDiffuse", 10);
        if (pointer != NULL) {
            pointer += 15;
            int num_chars = * ((int *) pointer);
            pointer += 4;
            memcpy(&buf[0], pointer, num_chars + 1);
            printf("texDiffuse = %d %s\n", num_chars, buf);
            texDiffuse[mat_index] = string(buf);
        }
    }
    pointer = material_block;
    mat_index = 0;
    while (pointer != NULL) {
        pointer = mem_find(pointer, size_material_block - (long)(pointer - material_block), "texSpecular", 11);
        if (pointer != NULL) {
            pointer += 16;
            int num_chars = *((int*)pointer);
            pointer += 4;
            memcpy(&buf[0], pointer, num_chars + 1);
            printf("texSpecular = %d %s\n", num_chars, buf);
            texSpecular[mat_index] = string(buf);
        }
    }
    delete[] material_block;
    // Далее идет блок координат и так далее
    // И снова перед нами полная жесть, много переменных неизвестного происхождения
    printf("ftell(f) = %x\n", ftell(f));
    fread(&parts_check, sizeof(int), 1, f);
    printf("parts_check = %08X\n", parts_check);
    fread(&n_parts, sizeof(int), 1, f);
    printf("N_parts = %d\n", n_parts);
    BMAGroup* groups = new BMAGroup[n_parts];
    for (unsigned int i = 0; i < n_parts; ++i) {
        if (version == 0x00040007 || version == 0x00040008) {
            fread(&groups[i].V_format, 4, 1, f);
            fread(&groups[i].material_id, 4, 1, f);
            fread(&groups[i].indeks_offset, 4, 1, f);
            fread(&groups[i].indeks_size, 4, 1, f);
            fread(&groups[i].vertex_offset, 4, 1, f);
            fread(&groups[i].bones_per_vertex, 4, 1, f);
            fread(&groups[i].bone_remap[0], 1, 16, f);
            printf("groups[%d].V_format = %d\n", i, groups[i].V_format);
            printf("groups[%d].material_id = %d\n", i, groups[i].material_id);
            printf("groups[%d].indeks_offset = %d\n", i, groups[i].indeks_offset);
            printf("groups[%d].indeks_size = %d\n", i, groups[i].indeks_size);
            printf("groups[%d].vertex_offset = %d\n", i, groups[i].vertex_offset);
            printf("groups[%d].bones_per_vertex = %d\n", i, groups[i].bones_per_vertex);
            printf("groups[%d].bones_remap = ", i);
            print_uarray(groups[i].bone_remap, 16);
            //printf("{%s", joints[groups[i].bone_remap[0]].name);
            //for (unsigned int j = 1; j < 16; ++j) {
            //    printf(", %s", joints[groups[i].bone_remap[j]].name);
            //}
            //printf("}\n");
            //printf("{%d", joints[groups[i].bone_remap[0]].unknown);
            //for (unsigned int j = 1; j < 16; ++j) {
            //    printf(", %d", joints[groups[i].bone_remap[j]].unknown);
            //}
            //printf("}\n");
        } else if (version == 0x00040006 || version == 0x00040005) {
            fread(&groups[i].V_format, 4, 1, f);
            fread(&groups[i].material_id, 4, 1, f);
            fread(&groups[i].indeks_offset, 4, 1, f);
            fread(&groups[i].indeks_size, 4, 1, f);
            fread(&groups[i].bones_per_vertex, 4, 1, f);
            fread(&groups[i].bone_remap[0], 1, 16, f);
            printf("groups[%d].V_format = %d\n", i, groups[i].V_format);
            printf("groups[%d].material_id = %d\n", i, groups[i].material_id);
            printf("groups[%d].indeks_offset = %d\n", i, groups[i].indeks_offset);
            printf("groups[%d].indeks_size = %d\n", i, groups[i].indeks_size);
            printf("groups[%d].bones_per_vertex = %d\n", i, groups[i].bones_per_vertex);
            printf("groups[%d].bones_remap = ", i);
            print_uarray(groups[i].bone_remap, 16);
            //printf("{%s", joints[groups[i].bone_remap[0]].name);
            //for (unsigned int j = 1; j < 16; ++j) {
            //    printf(", %s", joints[groups[i].bone_remap[j]].name);
            //}
            //printf("}\n");
            //printf("{%d", joints[groups[i].bone_remap[0]].unknown);
            //for (unsigned int j = 1; j < 16; ++j) {
            //    printf(", %d", joints[groups[i].bone_remap[j]].unknown);
            //}
            //printf("}\n");
        } else {
            printf("Error - unknown version of file type: %x", version);
            delete[] groups;
            fclose(f);
            return;
        }
    }
    printf("ftell(f) = %x\n", ftell(f));
    fread(&vertices_check, sizeof(int), 1, f);
    printf("vertices_check = %08X\n", vertices_check);
    fread(&n_vertices, sizeof(int), 1, f);
    printf("n_vertices = %d\n", n_vertices);
    vertices = new Vertex[n_vertices];
    VertexAdditional* vert_add = new VertexAdditional[n_vertices];
    // BMA файлы поддерживают только 256 костей и по 4 кости/веса на каждый вертекс
    if (groups[0].V_format == 3) {
        BMAFileFormat3* bmaformat3 = new BMAFileFormat3[n_vertices];
        fread(&bmaformat3[0], sizeof(BMAFileFormat3), n_vertices, f);
        if (bmaformat3[0].bm_scale == 0) {
            printf("Impossible! Str: %d\n", __LINE__);
        } else {
            for (unsigned int i = 0; i < n_vertices; ++i) {
                vertices[i].x = float(bmaformat3[i].bm_x) / bmaformat3[i].bm_scale;
                vertices[i].y = float(bmaformat3[i].bm_y) / bmaformat3[i].bm_scale;
                vertices[i].z = float(bmaformat3[i].bm_z) / bmaformat3[i].bm_scale;
                float nx = float((bmaformat3[i].bm_vector_n & 0x00FF0000) >> 16) / 256.0f;
                float ny = float((bmaformat3[i].bm_vector_n & 0x0000FF00) >> 8)  / 256.0f;
                float nz = float( bmaformat3[i].bm_vector_n & 0x000000FF)        / 256.0f;
                float nlength = sqrtf(nx * nx + ny * ny + nz * nz);
                vertices[i].nx = nx / nlength;
                vertices[i].ny = ny / nlength;
                vertices[i].nz = nz / nlength;
                vertices[i].u = float(bmaformat3[i].bm_u) / 2048.0f;
                vertices[i].v = float(bmaformat3[i].bm_v) / 2048.0f;
                float w1 = float(bmaformat3[i].weights[0]);
                float w2 = float(bmaformat3[i].weights[1]);
                float w3 = float(bmaformat3[i].weights[2]);
                float w4 = float(bmaformat3[i].weights[3]);
                float wsum = w1 + w2 + w3 + w4;
                //printf("BonesRemap[%d] = {%d, %d, %d, %d}\n", i, bmaformat3[i].bones[0], bmaformat3[i].bones[1], bmaformat3[i].bones[2], bmaformat3[i].bones[3]);
                vertices[i].weights[0] = w1 / wsum;
                vertices[i].weights[1] = w2 / wsum;
                vertices[i].weights[2] = w3 / wsum;
                vertices[i].weights[3] = 1.0f - (w1 + w2 + w3) / wsum;
                vert_add[i].bones_remap[0] = (bmaformat3[i].bones[0] != 255) ? bmaformat3[i].bones[0] : 0;
                vert_add[i].bones_remap[1] = (bmaformat3[i].bones[1] != 255) ? bmaformat3[i].bones[1] : 0;
                vert_add[i].bones_remap[2] = (bmaformat3[i].bones[2] != 255) ? bmaformat3[i].bones[2] : 0;
                vert_add[i].bones_remap[3] = (bmaformat3[i].bones[3] != 255) ? bmaformat3[i].bones[3] : 0;
                //printf("Weights[%d] = {%f, %f, %f, %f}\n", i, w1, w2, w3, w4);
                //printf("Bones[%d] = {%d, %d, %d, %d} PartIndex = %d\n", i, vertices[i].bones[0], vertices[i].bones[1], vertices[i].bones[2], vertices[i].bones[3], part_index);
            }
        }
        delete [] bmaformat3;
    } else if (groups[0].V_format == 7) {
        BMAFileFormat7* bmaformat7 = new BMAFileFormat7[n_vertices];
        fread(&bmaformat7[0], sizeof(BMAFileFormat7), n_vertices, f);
        if (bmaformat7[0].bm_scale == 0) {
            printf("Impossible! Str: %d\n", __LINE__);
        } else {
            for (unsigned int i = 0; i < n_vertices; ++i) {
                vertices[i].x = float(bmaformat7[i].bm_x) / bmaformat7[i].bm_scale;
                vertices[i].y = float(bmaformat7[i].bm_y) / bmaformat7[i].bm_scale;
                vertices[i].z = float(bmaformat7[i].bm_z) / bmaformat7[i].bm_scale;
                float nx = float((bmaformat7[i].bm_vector_n & 0x00FF0000) >> 16) / 256.0f;
                float ny = float((bmaformat7[i].bm_vector_n & 0x0000FF00) >> 8)  / 256.0f;
                float nz = float( bmaformat7[i].bm_vector_n & 0x000000FF)        / 256.0f;
                float nlength = sqrtf(nx * nx + ny * ny +nz * nz);
                vertices[i].nx = nx/nlength;
                vertices[i].ny = ny/nlength;
                vertices[i].nz = nz/nlength;
                vertices[i].u = float(bmaformat7[i].bm_u) / 2048.0f;
                vertices[i].v = float(bmaformat7[i].bm_v) / 2048.0f;
                float w1 = float(bmaformat7[i].weights[0]);
                float w2 = float(bmaformat7[i].weights[1]);
                float w3 = float(bmaformat7[i].weights[2]);
                float w4 = float(bmaformat7[i].weights[3]);
                float wsum = w1 + w2 + w3 + w4;
                vertices[i].weights[0] = w1/wsum;
                vertices[i].weights[1] = w2/wsum;
                vertices[i].weights[2] = w3/wsum;
                vertices[i].weights[3] = 1.0f - (w1 + w2 + w3) / wsum;
                vert_add[i].bones_remap[0] = (bmaformat7[i].bones[0] != 255) ? bmaformat7[i].bones[0] : 0;
                vert_add[i].bones_remap[1] = (bmaformat7[i].bones[1] != 255) ? bmaformat7[i].bones[1] : 0;
                vert_add[i].bones_remap[2] = (bmaformat7[i].bones[2] != 255) ? bmaformat7[i].bones[2] : 0;
                vert_add[i].bones_remap[3] = (bmaformat7[i].bones[3] != 255) ? bmaformat7[i].bones[3] : 0;
            }
        }
        delete[] bmaformat7;
    } else {
        printf("Unknown vertex format: %d\n", groups[0].V_format);
        fclose(f);
        delete[] groups;
        delete[] vertices;
        return;
    }
    printf("ftell(f) = %x\n", ftell(f));
    fread(&indices_check, sizeof(int), 1, f);
    printf("indices_check = %08X\n", indices_check);
    fread(&n_indices, sizeof(int), 1, f);
    printf("n_indices = %d\n", n_indices);
    indices = new unsigned short[n_indices];
    fread(&indices[0], sizeof(unsigned short), n_indices, f);
    printf("ftell(f) = %x\n", ftell(f));
    unsigned int part_index = 0;
    for (unsigned int i = 0; i < n_indices; ++i) {
        if (vert_add[indices[i]].parts.find(part_index) != vert_add[indices[i]].parts.end()) {
            vert_add[indices[i]].parts[part_index]++;   
        } else {
            vert_add[indices[i]].parts.insert({part_index, 0});
        }
        if ((i + 1) >= (groups[part_index].indeks_offset + groups[part_index].indeks_size)) {
            //printf("Part block = %d\n", i + 1);
            part_index++;
            if (part_index > n_parts) {
                printf("Something wrong, if part_index is not last! part_index = %d\n", part_index);
            }
        }
    }
    for (unsigned int i = 0; i<n_vertices; ++i) {
        if (vert_add[i].parts.size() > 0) {
            int part = vert_add[i].parts.begin()->first;
            vertices[i].bones[0] = (vertices[i].weights[0] > 0.0f) ? groups[part].bone_remap[vert_add[i].bones_remap[0]] : 0;
            vertices[i].bones[1] = (vertices[i].weights[1] > 0.0f) ? groups[part].bone_remap[vert_add[i].bones_remap[1]] : 0;
            vertices[i].bones[2] = (vertices[i].weights[2] > 0.0f) ? groups[part].bone_remap[vert_add[i].bones_remap[2]] : 0;
            vertices[i].bones[3] = (vertices[i].weights[3] > 0.0f) ? groups[part].bone_remap[vert_add[i].bones_remap[3]] : 0;
        } else {
            vertices[i].bones[0] = 0;
            vertices[i].bones[1] = 0;
            vertices[i].bones[2] = 0;
            vertices[i].bones[3] = 0;
        }
        //float w = vertices[i].weights[0] + vertices[i].weights[1] + vertices[i].weights[2] + vertices[i].weights[3];
        //printf("Weights[%d] = {%f, %f, %f, %f} sum = %f\n", i, vertices[i].weights[0], vertices[i].weights[1], vertices[i].weights[2], vertices[i].weights[3], w);
        //printf("Bones[%d] = {%d, %d, %d, %d} PartIndex = %d\n", i, vertices[i].bones[0], vertices[i].bones[1], vertices[i].bones[2], vertices[i].bones[3], part);
        if (vert_add[i].parts.size() > 1) {
            printf("vertices[%d].parts = { ", i);
            for (auto part : vert_add[i].parts) {
                printf("{%d, %d} ", part.first, part.second);
            }
            printf(" }\n");
        }
    }
    minVertex = vertices[0];
    maxVertex = vertices[0];
    for (unsigned int i = 1; i < n_vertices; ++i) {
        if (vertices[i].x < minVertex.x) minVertex.x = vertices[i].x;
        if (vertices[i].y < minVertex.y) minVertex.y = vertices[i].y;
        if (vertices[i].z < minVertex.z) minVertex.z = vertices[i].z;
        if (vertices[i].nx < minVertex.nx) minVertex.nx = vertices[i].nx;
        if (vertices[i].ny < minVertex.ny) minVertex.ny = vertices[i].ny;
        if (vertices[i].nz < minVertex.nz) minVertex.nz = vertices[i].nz;
        if (vertices[i].u < minVertex.u) minVertex.u = vertices[i].u;
        if (vertices[i].v < minVertex.v) minVertex.v = vertices[i].v;
        if (vertices[i].bones[0] < minVertex.bones[0]) minVertex.bones[0] = vertices[i].bones[0];
        if (vertices[i].bones[1] < minVertex.bones[1]) minVertex.bones[1] = vertices[i].bones[1];
        if (vertices[i].bones[2] < minVertex.bones[2]) minVertex.bones[2] = vertices[i].bones[2];
        if (vertices[i].bones[3] < minVertex.bones[3]) minVertex.bones[3] = vertices[i].bones[3];
        if (vertices[i].weights[0] < minVertex.weights[0]) minVertex.weights[0] = vertices[i].weights[0];
        if (vertices[i].weights[1] < minVertex.weights[1]) minVertex.weights[1] = vertices[i].weights[1];
        if (vertices[i].weights[2] < minVertex.weights[2]) minVertex.weights[2] = vertices[i].weights[2];
        if (vertices[i].weights[3] < minVertex.weights[3]) minVertex.weights[3] = vertices[i].weights[3];

        if (vertices[i].x > maxVertex.x) maxVertex.x = vertices[i].x;
        if (vertices[i].y > maxVertex.y) maxVertex.y = vertices[i].y;
        if (vertices[i].z > maxVertex.z) maxVertex.z = vertices[i].z;
        if (vertices[i].nx > maxVertex.nx) maxVertex.nx = vertices[i].nx;
        if (vertices[i].ny > maxVertex.ny) maxVertex.ny = vertices[i].ny;
        if (vertices[i].nz > maxVertex.nz) maxVertex.nz = vertices[i].nz;
        if (vertices[i].u > maxVertex.u) maxVertex.u = vertices[i].u;
        if (vertices[i].v > maxVertex.v) maxVertex.v = vertices[i].v;
        if (vertices[i].bones[0] > maxVertex.bones[0]) maxVertex.bones[0] = vertices[i].bones[0];
        if (vertices[i].bones[1] > maxVertex.bones[1]) maxVertex.bones[1] = vertices[i].bones[1];
        if (vertices[i].bones[2] > maxVertex.bones[2]) maxVertex.bones[2] = vertices[i].bones[2];
        if (vertices[i].bones[3] > maxVertex.bones[3]) maxVertex.bones[3] = vertices[i].bones[3];
        if (vertices[i].weights[0] > maxVertex.weights[0]) maxVertex.weights[0] = vertices[i].weights[0];
        if (vertices[i].weights[1] > maxVertex.weights[1]) maxVertex.weights[1] = vertices[i].weights[1];
        if (vertices[i].weights[2] > maxVertex.weights[2]) maxVertex.weights[2] = vertices[i].weights[2];
        if (vertices[i].weights[3] > maxVertex.weights[3]) maxVertex.weights[3] = vertices[i].weights[3];
    }
    delete[] vert_add;
    delete[] groups;
    fclose(f);
}

KBParser::~KBParser() {
    delete[] vertices;
    delete[] indices;
    delete[] joints;
}

void KBParser::toGLTF(const string& filename, const string& bsa_path, const string& texture_path, const string& out_path, const string& prefix) {
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Подготовка данных завершена, можно заполнять данные gltf
    ////////////////////////////////////////////////////////////////////////////////////////////////
    tinygltf::Model m;      // Файл gltf
    tinygltf::Scene scene;  // Сцена
    // Непонятная штука, видимо задает фичи gltf
    m.asset.version = "2.0";
    m.asset.generator = "tinygltf";
    m.extensionsUsed.push_back("KHR_materials_specular");
    tinygltf::Mesh mesh;  // Меш в сцене
    // В каждом gltf mesh есть primitive (определяет как мы рисуем треугольники) он может быть и не один
    tinygltf::Primitive primitive;
    primitive.indices = 1;
    primitive.attributes["POSITION"] = 0;
    primitive.attributes["NORMAL"] = 2;
    primitive.attributes["TEXCOORD_0"] = 3;
    primitive.attributes["JOINTS_0"] = 4;
    primitive.attributes["WEIGHTS_0"] = 5;
    primitive.material = 0;
    primitive.mode = TINYGLTF_MODE_TRIANGLES;

    mesh.primitives.push_back(primitive);

    m.meshes.push_back(mesh);
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Буферы
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Вертексные буферы
    tinygltf::Buffer b_vertices;  // Будет содержать все данные вертексов
    tinygltf::Buffer b_triangles; // Все индексы
    tinygltf::BufferView bv_vertices;  // Общий bufferView c byteStryde = sizeof(Vertex)
    tinygltf::BufferView bv_triangles; // Общий bufferView c byteStryde = sizeof(Vertex)
    tinygltf::Accessor a_positions;    // Позиции
    tinygltf::Accessor a_triangles;    // Индексы
    tinygltf::Accessor a_normals;      // Нормали
    tinygltf::Accessor a_uvs;          // UV
    tinygltf::Accessor a_joints;       // Кости
    tinygltf::Accessor a_weights;      // Веса
    
    b_vertices.data.resize(n_vertices * sizeof(Vertex));
    memcpy(&b_vertices.data[0], &vertices[0], n_vertices * sizeof(Vertex));
    b_triangles.data.resize(n_indices * sizeof(unsigned short));
    memcpy(&b_triangles.data[0], &indices[0], n_indices * sizeof(unsigned short));

    m.buffers.push_back(b_vertices);
    m.buffers.push_back(b_triangles);

    bv_vertices.buffer = 0;
    bv_vertices.byteOffset = 0;
    bv_vertices.byteStride = sizeof(Vertex);
    bv_vertices.byteLength = n_vertices * sizeof(Vertex);
    bv_vertices.target = TINYGLTF_TARGET_ARRAY_BUFFER;

    a_positions.bufferView = 0;
    a_positions.byteOffset = 0;
    a_positions.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    a_positions.count = n_vertices;
    a_positions.type = TINYGLTF_TYPE_VEC3;
    a_positions.maxValues = {maxVertex.x, maxVertex.y, maxVertex.z};
    a_positions.minValues = {minVertex.x, minVertex.y, minVertex.z};

    a_normals.bufferView = 0;
    a_normals.byteOffset = 12;
    a_normals.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    a_normals.count = n_vertices;
    a_normals.type = TINYGLTF_TYPE_VEC3;
    a_normals.maxValues = {maxVertex.nx, maxVertex.ny, maxVertex.nz};
    a_normals.minValues = {minVertex.nx, minVertex.ny, minVertex.nz};

    a_uvs.bufferView = 0;
    a_uvs.byteOffset = 24;
    a_uvs.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    a_uvs.count = n_vertices;
    a_uvs.type = TINYGLTF_TYPE_VEC2;
    a_uvs.maxValues = {maxVertex.u, maxVertex.v};
    a_uvs.minValues = {minVertex.u, minVertex.v};

    a_joints.bufferView = 0;
    a_joints.byteOffset = 32;
    a_joints.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
    a_joints.count = n_vertices;
    a_joints.type = TINYGLTF_TYPE_VEC4;
    a_joints.maxValues = {(double)maxVertex.bones[0], (double)maxVertex.bones[1], (double)maxVertex.bones[2], (double)maxVertex.bones[3]};
    a_joints.minValues = {(double)minVertex.bones[0], (double)minVertex.bones[1], (double)minVertex.bones[2], (double)minVertex.bones[3]};

    a_weights.bufferView = 0;
    a_weights.byteOffset = 36;
    a_weights.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    a_weights.count = n_vertices;
    a_weights.type = TINYGLTF_TYPE_VEC4;
    a_weights.maxValues = {(double)maxVertex.weights[0], (double)maxVertex.weights[1], (double)maxVertex.weights[2], (double)maxVertex.weights[3]};
    a_weights.minValues = {(double)minVertex.weights[0], (double)minVertex.weights[1], (double)minVertex.weights[2], (double)minVertex.weights[3]};

    bv_triangles.buffer = 1;
    bv_triangles.byteOffset = 0;
    bv_triangles.byteLength = n_indices * sizeof(unsigned short);
    bv_triangles.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;

    a_triangles.bufferView = 1;
    a_triangles.byteOffset = 0;
    a_triangles.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT;
    a_triangles.count = n_indices;
    a_triangles.type = TINYGLTF_TYPE_SCALAR;
    a_triangles.maxValues.push_back(n_vertices - 1);
    a_triangles.minValues.push_back(0);

    m.bufferViews.push_back(bv_vertices);
    m.bufferViews.push_back(bv_triangles);

    m.accessors.push_back(a_positions);  // 0
    m.accessors.push_back(a_triangles);  // 1
    m.accessors.push_back(a_normals);    // 2
    m.accessors.push_back(a_uvs);        // 3
    m.accessors.push_back(a_joints);     // 4
    m.accessors.push_back(a_weights);    // 5

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Отдельная проблема это InverseBindMatrices, благо они здесь посчитаны
    ///////////////////////////////////////////////////////////////////////////////////////////////
    tinygltf::Buffer ibm_buffer;
    tinygltf::BufferView ibm_buffer_view;
    tinygltf::Accessor ibm_accessor;

    float* ibm = new float[16 * n_joints];
    float inv_ibm[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    for (unsigned int i = 0; i < n_joints; ++i) {
        joints[i].matrix[15] = 1.0f;
        memcpy(&ibm[16 * i], &joints[i].matrix[0], 16 * sizeof(float));
    }
    ibm_buffer.data.resize(16 * n_joints * sizeof(float));
    memcpy(&ibm_buffer.data[0], &ibm[0], 16 * n_joints * sizeof(float));
    m.buffers.push_back(ibm_buffer);

    ibm_buffer_view.buffer = (unsigned)m.buffers.size() - 1;
    ibm_buffer_view.byteOffset = 0;
    ibm_buffer_view.byteLength = 16 * n_joints * sizeof(float);
    m.bufferViews.push_back(ibm_buffer_view);

    ibm_accessor.bufferView = (unsigned)m.bufferViews.size() - 1;
    ibm_accessor.byteOffset = 0;
    ibm_accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    ibm_accessor.count = n_joints;
    ibm_accessor.type = TINYGLTF_TYPE_MAT4;
    m.accessors.push_back(ibm_accessor);

    unsigned ibm_accessor_index = (unsigned)m.accessors.size() - 1;
    delete[] ibm;
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Скелет
    ///////////////////////////////////////////////////////////////////////////////////////////////
    
    for (unsigned int i = 0; i < n_joints; ++i) {
        tinygltf::Node node;  // Структура скелета передаётся в node (пока не используется)
        if (i > 0) {
            glm::mat4 ibmg = glm::make_mat4(joints[i].matrix);
            glm::mat4 pibmg = glm::make_mat4(joints[joints[i].parent].matrix);
            float x, y, z, a, px, py, pz;
            calc_quaternion(pibmg*glm::inverse(ibmg), x, y, z, a, px, py, pz);
            node.rotation = {x, y, z, a};
            node.translation = {px, py, pz};
        } else {
            node.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
            node.translation = {0.0f, 0.0f, 0.0f};
        }
        node.name = string(joints[i].name);
        m.nodes.push_back(node);
    }
    for (unsigned int i = 1; i < n_joints; ++i) {
        m.nodes[joints[i].parent].children.push_back(i);
    }
    m.nodes[0].mesh = 0;
    m.nodes[0].skin = 0;
    
    tinygltf::Skin skin;
    for (unsigned int i = 0; i < n_joints; ++i) {
        skin.joints.push_back(i);
    }
    skin.inverseBindMatrices = ibm_accessor_index;
    m.skins.push_back(skin);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Анимации
    ///////////////////////////////////////////////////////////////////////////////////////////////
     
    std::string bsa_ext = ".bsa";
    for (const auto& file : fs::directory_iterator(bsa_path)) {
        if (file.path().extension() == bsa_ext) {
            //std::cout << file.path().stem() << std::endl;
            string wext = file.path().stem().string();
            if (wext.starts_with(filename + "_")) {
                BSAParser bsa_parser = BSAParser(file.path().string());
                string anim_name = wext.substr(filename.size() + 1);
                cout << "Animation name = " << anim_name << endl;
                bsa_parser.toGLTF(m, anim_name);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Материалы
    ///////////////////////////////////////////////////////////////////////////////////////////////

    tinygltf::Material mat;
    mat.doubleSided = true;
    mat.alphaMode = "MASK";
    string diffuse_texture = texture_path + texDiffuse[0];
    string spec_texture = texture_path +texSpecular[0];
    string diffuse_texture2 = "textures\\" + prefix + texDiffuse[0];
    string spec_texture2 = "textures\\" + prefix + texSpecular[0];
    const auto copy_options = fs::copy_options::update_existing;
    if (fs::exists(diffuse_texture)) {
        printf("DiffuseTexture: %s\n", diffuse_texture2.c_str());
        fs::copy(diffuse_texture, out_path + diffuse_texture2, copy_options);
        tinygltf::Texture tex_color;
        tex_color.sampler = 0;
        tex_color.source = 0;
        m.textures.push_back(tex_color);
    
        tinygltf::Image img_color;
        img_color.uri = diffuse_texture2;
        m.images.push_back(img_color);

        mat.pbrMetallicRoughness.baseColorTexture.index = (unsigned)m.textures.size() - 1;
    }
    if (fs::exists(spec_texture)) {
        printf("SpecularTexture: %s\n", spec_texture2.c_str());
        fs::copy(spec_texture, out_path + spec_texture2, copy_options);
        tinygltf::Texture tex_color;
        tex_color.sampler = 0;
        tex_color.source = 1;
        m.textures.push_back(tex_color);

        tinygltf::Image img_color;
        img_color.uri = spec_texture2;
        m.images.push_back(img_color);

        
        tinygltf::Value::Object khr_spec_texture_index;
        khr_spec_texture_index.insert(
            {"index", tinygltf::Value(1)}
        );
        tinygltf::Value::Object khr_spec_texture;
        khr_spec_texture.insert(
            {"specularTexture", tinygltf::Value(khr_spec_texture_index)}
        );
        mat.extensions.insert(
            {{"KHR_materials_specular", tinygltf::Value(khr_spec_texture)}}
        );
    }
    m.materials.push_back(mat);
    
    if (m.samplers.size() == 0) {
        tinygltf::Sampler smp;
        smp.magFilter = TINYGLTF_TEXTURE_FILTER_LINEAR;
        smp.minFilter = TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR;
        smp.wrapS = TINYGLTF_TEXTURE_WRAP_REPEAT;
        smp.wrapT = TINYGLTF_TEXTURE_WRAP_REPEAT;
        m.samplers.push_back(smp);
    }

    //
    tinygltf::TinyGLTF gltf;
    scene.nodes.push_back(0);
    m.scenes.push_back(scene);

    gltf.WriteGltfSceneToFile(&m, out_path + prefix + filename + ".gltf",
                              true,    // embedImages
                              true,    // embedBuffers
                              true,    // pretty print
                              false);  // write binary
}

int main() {
    if constexpr (std::endian::native == std::endian::little)
        std::cout << "little-endian\n";
    else if constexpr (std::endian::native == std::endian::big)
        std::cout << "big-endian\n";
    else
        std::cout << "mixed-endian\n";
    vector<string> filenames;
    //= {
    //    "berserkthick"
    //};
    std::string bsa_path = "C:\\All\\All\\Games\\King's Bounty - Warriors of the North\\data\\animation\\";
    std::string texture_path = "C:\\All\\All\\Games\\King's Bounty - Warriors of the North\\data\\textures\\";
    std::string bma_path = "C:\\All\\All\\Games\\King's Bounty - Warriors of the North\\data\\models\\";
    std::string bma_ext = ".bma";
    std::string out_path = "C:\\All\\All\\GameDev\\models\\";
    std::string prefix = "kbw_"; // kbw - woth, kbl - legend, kbp - princess in armor, kbd - darkside
    for (const auto& file : fs::directory_iterator(bma_path)) {
        if (file.path().extension() == bma_ext) {
            std::cout << file.path().stem() << std::endl;
            filenames.push_back(file.path().stem().string());
        }
    }
    for (auto iter = filenames.begin(); iter != filenames.end(); iter++) {
        std::cout << *iter << std::endl;
        try {
            KBParser kb = KBParser(*iter + ".bma", bma_path);
            kb.toGLTF(*iter, bsa_path, texture_path, out_path, prefix);
        } catch (...) {
            std::cout << "Error!" << std::endl;
        }
    }
    return 0;
}
