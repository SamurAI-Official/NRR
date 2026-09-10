#include "nrr_temporal.h"
#include <iostream>
#include <vector>

void print_state(const NRRTemporalState& s) {
    std::cout << "  Frame: " << s.frame_index << ", Delta: " << s.delta_time << "s" << std::endl;
    std::cout << "  Resolution: " << s.resolution_x << "x" << s.resolution_y << std::endl;
    std::cout << "  Motion: " << s.motion_magnitude << ", Alpha: " << s.temporal_alpha << std::endl;
    std::cout << "  History Frames: " << s.history_frames << std::endl;
}

int main() {
    std::cout << "=== NRR Temporal Rendering Test ===" << std::endl;

    nrr::TemporalHistory history;
    history.set_max_frames(4);

    for (uint32_t i = 0; i < 6; i++) {
        std::vector<float> c(10000 * 3, static_cast<float>(i) * 0.1f);
        std::vector<float> d(10000, static_cast<float>(i) * 0.01f);
        std::vector<float> m(10000 * 2, 0.0f);
        history.add_frame(i, i * 0.016f, c, d, m, 100, 100, NRR_TEXTURE_FORMAT_RGB8, NRR_TEXTURE_FORMAT_R32F);
    }

    std::cout << "Added 6 frames, history: " << history.get_frame_count() << std::endl;
    std::cout << "Max frames: " << history.get_max_frames() << std::endl;

    std::vector<float> pc, pd, pm;
    uint32_t pw, ph;
    bool got = history.get_previous_frame(6, pc, pd, pm, pw, ph);
    std::cout << "Got previous: " << (got ? "yes" : "no") << ", size: " << pw << "x" << ph << std::endl;
    std::cout << std::endl;

    nrr::TemporalStateManager mgr;
    NRRDevice* dev = nullptr;
    mgr.initialize(dev);

    NRRFrameInput in = {};
    in.temporal.frame_index = 1;
    in.temporal.delta_time = 0.016f;
    in.temporal.resolution_x = 1920;
    in.temporal.resolution_y = 1080;
    in.temporal.motion_magnitude = 0.1f;
    in.temporal.temporal_alpha = 0.0f;
    in.temporal.history_frames = 0;

    NRRFrameOutput out = {};
    out.temporal = in.temporal;

    auto st = mgr.update_state(dev, in, out, history);
    std::cout << "Frame 1 (low motion):" << std::endl;
    print_state(st);
    std::cout << std::endl;

    in.temporal.frame_index = 2;
    st = mgr.update_state(dev, in, out, history);
    std::cout << "Frame 2 (low motion):" << std::endl;
    print_state(st);
    std::cout << std::endl;

    in.temporal.frame_index = 3;
    in.temporal.motion_magnitude = 0.8f;
    st = mgr.update_state(dev, in, out, history);
    std::cout << "Frame 3 (high motion):" << std::endl;
    print_state(st);
    std::cout << std::endl;

    nrr::TemporalRenderer renderer;
    renderer.initialize(dev);

    nrr::HistoryEntry prev;
    prev.frame_index = 1;
    prev.width = 100; prev.height = 100;
    prev.color_data = std::vector<float>(30000, 0.5f);
    prev.depth_data = std::vector<float>(10000, 0.5f);
    prev.motion_data = std::vector<float>(20000, 1.0f);

    std::vector<float> wc, wd;
    bool warped = renderer.warp_previous_output(in, prev, wc, wd);
    std::cout << "Warped: " << (warped ? "success" : "failed") << std::endl;
    std::cout << "  Color size: " << wc.size() << ", Depth size: " << wd.size() << std::endl;
    std::cout << std::endl;

    float stab = renderer.calculate_temporal_stability(out, out);
    std::cout << "Stability: " << stab << std::endl;
    std::cout << std::endl;

    mgr.reset();
    history.clear();
    st = mgr.update_state(dev, in, out, history);
    std::cout << "After reset:" << std::endl;
    print_state(st);
    std::cout << std::endl;

    mgr.shutdown();
    renderer.shutdown();

    std::cout << "=== Test Complete ===" << std::endl;
    return 0;
}