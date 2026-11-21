// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/inkling2.hpp"
#include "b70/json.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>

using namespace b70::inkling2;

int main(int argc,char** argv) {
    const auto check_bad=[](auto fn){
        try { fn(); }
        catch(const std::exception&) { return; }
        throw std::runtime_error("expected rejection");
    };
    if(argc>=2) {
        std::ifstream in(argv[1]);
        std::string raw(std::istreambuf_iterator<char>(in),{});
        auto c=parse_config(raw);
        assert(c.layers==24 && c.cache_slots()==48 && c.output_vocab==200058 && c.vocab==201024);
        const auto v=c.visits();
        int full=0,revisit=0;
        std::set<int> slots;
        for(auto visit:v) {
            full+=!visit.sliding; revisit+=visit.has_message;
            assert(slots.insert(visit.cache_slot).second);
        }
        assert(v.size()==48 && full==8 && revisit==24);
        assert(v[6].physical==6 && v[6].cache_slot==6 && !v[6].has_message);
        assert(v[18].physical==6 && v[18].cache_slot==24 && v[18].has_message);
        assert(v[30].physical==6 && v[30].cache_slot==36 && v[30].has_message);
        assert(v[42].physical==18 && v[42].cache_slot==18 && !v[42].has_message);
        auto cfg=b70::Json::parse(raw);
        cfg.object["text_config"].object["loop_config"].object["is_varloop"].boolean=true;
        check_bad([&]{parse_config(cfg.dump());});
    }
    // Bias changes selection, but normalized weights use the original sigmoid scores.
    const auto r=route(std::array{0.f,1.f,-1.f},std::array{0.f,0.f,1.f},2,8.f);
    const float a=1.f/(1.f+std::exp(1.f)),b=1.f/(1.f+std::exp(-1.f));
    assert(r.size()==2 && r[0].expert==2 && r[1].expert==1);
    assert(std::abs(r[0].weight-8.f*a/(a+b))<1e-6f);
    assert(std::abs(r[1].weight-8.f*b/(a+b))<1e-6f);
    const float g=6.f,u=-3.f;
    const float expected=4.f*std::tanh(g/4.f)/(1.f+std::exp(-g))*25.f*std::tanh(u/25.f);
    assert(std::abs(swiglu(g,u,4.f,25.f)-expected)<2e-6f);
    // Codes 2 and 11 are +1 and -1.5, per-expert multiplier amax/denom=2/4.
    std::array<uint8_t,16> fp4{}; fp4.fill(0xb2);
    std::array<uint8_t,2> scales{0x40,0x38}; // E4M3 2, 1
    std::array<float,2> amax{2.f,8.f};
    PackedRows nv{PackedFormat::NVFP4,2,1,16,fp4.data(),scales.data(),amax.data(),4.f};
    std::array<float,16> row;
    nv.row(0,0,row); assert(row[0]==1.f && row[1]==-1.5f);
    nv.row(1,0,row); assert(row[0]==2.f && row[1]==-3.f);
    std::array<uint8_t,32> fp8{}; fp8.fill(0x38);
    const uint8_t adjusted=126;
    PackedRows mx{PackedFormat::MXFP8,1,1,32,fp8.data(),&adjusted};
    std::array<float,32> widened;
    mx.row(0,0,widened); assert(widened[0]==.5f && widened[31]==.5f);
    nv.denominator=0;
    check_bad([&]{nv.row(0,0,row);});
    auto one=read_tensor_header(R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})");
    assert(one.at("x").shape==std::vector<int64_t>{1});
    check_bad([]{read_tensor_header(R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,3]}})");});
    check_bad([]{read_tensor_header(R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})");});
    if(argc>=3) {
        auto c=Checkpoint::open(argv[2]);
        TextReference model(std::move(c),16);
        std::vector<float> first(model.config().output_vocab), after(first.size());
        const int id=model.step(2,first);
        model.step(7);
        model.step(9);
        check_bad([&]{model.step(model.config().output_vocab);});  // padded ID cannot advance the sequence
        assert(model.position()==3);
        model.reset();
        const int again=model.step(2,after);
        assert(model.position()==1 && id==again && first==after);
    }
    std::cout<<"PASS Inkling2 loop/cache schedule, packed weights, routing, SwiGLU, reset and malformed header rejection\n";
}
