// ---------------------------------------------------------------------------
// NRRBlit.shader
// Copies a source texture to the target. The source is the NRR session's output
// and the target is the camera colour, so the copy *is* the composite.
//
// This is the URP 17 (Unity 6) convention rather than the URP 12 one. Under
// RenderGraph the blit binds its source as _BlitTexture - the default property in
// RenderGraphUtils.BlitMaterialParameters - and draws a full-screen triangle, so
// the vertex work and the scale-bias uv come from the core package's Blit.hlsl
// instead of from a quad whose source arrived as _MainTex.
// ---------------------------------------------------------------------------
Shader "Hidden/NRR/Blit"
{
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" }
        LOD 100
        ZWrite Off Cull Off

        Pass
        {
            Name "NRRBlit"
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"

            half4 Frag(Varyings input) : SV_Target
            {
                return SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, input.texcoord);
            }
            ENDHLSL
        }
    }
}
