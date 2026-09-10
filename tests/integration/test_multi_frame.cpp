#include "test_framework.h"
#include "nrr.h"
#include <vector>

namespace nrr {
namespace test {

class TemporalHistory {
public:
    void set_max_frames(uint32_t max_frames) {
        m_max_frames = max_frames;
    }
    
    uint32_t get_frame_count() const {
        return std::min(m_frames.size(), (size_t)m_max_frames);
    }
    
    void add_frame(uint32_t frame_index, float delta_time,
                   const std::vector<float>& color,
                   const std::vector<float>& depth,
                   const std::vector<float>& motion,
                   uint32_t width, uint32_t height,
                   NRRTextureFormat color_format,
                   NRRTextureFormat depth_format) {
        FrameData frame;
        frame.frame_index = frame_index;
        frame.delta_time = delta_time;
        frame.color_data = color;
        frame.depth_data = depth;
        frame.motion_data = motion;
        frame.width = width;
        frame.height = height;
        frame.color_format = color_format;
        frame.depth_format = depth_format;
        m_frames.push_back(frame);
        
        if (m_frames.size() > m_max_frames) {
            m_frames.erase(m_frames.begin());
        }
    }
    
    bool get_previous_frame(uint32_t current_frame,
                           std::vector<float>& out_color,
                           std::vector<float>& out_depth,
                           std::vector<float>& out_motion,
                           uint32_t& out_w, uint32_t& out_h) {
        for (int i = (int)m_frames.size() - 1; i >= 0; i--) {
            if (m_frames[i].frame_index < current_frame && m_frames[i].frame_index > 0) {
                out_color = m_frames[i].color_data;
                out_depth = m_frames[i].depth_data;
                out_motion = m_frames[i].motion_data;
                out_w = m_frames[i].width;
                out_h = m_frames[i].height;
                return true;
            }
        }
        return false;
    }
    
    float calc_motion_mag(uint32_t frame_index,
                          const std::vector<float>& motion,
                          uint32_t width, uint32_t height) {
        if (motion.empty()) return 0.0f;
        
        float sum = 0.0f;
        size_t count = motion.size() / 2;
        for (size_t i = 0; i < count; i++) {
            float mx = motion[i * 2];
            float my = motion[i * 2 + 1];
            sum += std::sqrt(mx * mx + my * my);
        }
        
        float avg = sum / std::max(count, (size_t)1);
        return std::min(avg / 10.0f, 1.0f);
    }
    
private:
    struct FrameData {
        uint32_t frame_index;
        float delta_time;
        std::vector<float> color_data;
        std::vector<float> depth_data;
        std::vector<float> motion_data;
        uint32_t width, height;
        NRRTextureFormat color_format;
        NRRTextureFormat depth_format;
    };
    
    std::vector<FrameData> m_frames;
    uint32_t m_max_frames = 4;
};

NRR_TEST(test_temporal_history_buffer) {
    TemporalHistory history;
    history.set_max_frames(4);
    
    for (uint32_t i = 0; i < 6; i++) {
        std::vector<float> color(10000, static_cast<float>(i) * 0.1f);
        std::vector<float> depth(10000, static_cast<float>(i) * 0.01f);
        std::vector<float> motion(20000, 0.0f);
        
        history.add_frame(i, i * 0.016f, color, depth, motion, 100, 100,
                         NRR_TEXTURE_FORMAT_RGBA8, NRR_TEXTURE_FORMAT_R32F);
    }
    
    NRR_EXPECT_EQ(history.get_frame_count(), 4, "History should be capped at 4 frames");
    
    std::vector<float> prev_color, prev_depth, prev_motion;
    uint32_t pw, ph;
    bool got = history.get_previous_frame(6, prev_color, prev_depth, prev_motion, pw, ph);
    
    NRR_EXPECT_TRUE(got, "Should get previous frame");
    NRR_EXPECT_EQ(pw, 100, "Width should be 100");
    NRR_EXPECT_EQ(ph, 100, "Height should be 100");
    
    std::cout << "  History size: " << history.get_frame_count() << std::endl;
    std::cout << "  Previous frame: " << pw << "x" << ph << std::endl;
}

NRR_TEST(test_motion_magnitude_calculation) {
    TemporalHistory history;
    
    std::vector<float> motion(10000 * 2);
    for (size_t i = 0; i < 10000; i++) {
        motion[i * 2] = 2.0f;
        motion[i * 2 + 1] = 0.0f;
    }
    
    float magnitude = history.calc_motion_mag(1, motion, 100, 100);
    
    NRR_EXPECT_NEAR(magnitude, 0.2f, 0.01f, "Motion magnitude should be ~0.2");
    
    std::cout << "  Motion magnitude: " << magnitude << std::endl;
}

} // namespace test
} // namespace nrr