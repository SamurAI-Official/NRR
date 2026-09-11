// ---------------------------------------------------------------------------
// NRRModelManagerWindow.cs
// Editor tool for managing NRR devices, models and references from the Unity
// editor. Opens via Window > NRR > Model Manager.
// ---------------------------------------------------------------------------
using System;
using UnityEditor;
using UnityEngine;

namespace NRR.Editor
{
    public class NRRModelManagerWindow : EditorWindow
    {
        private NRRSettings _settings;
        private string _status = "Not initialized.";

        private NRRDevice _device;
        private NRRModel _model;
        private NRRReference _reference;
        private NRRCapabilities _caps;
        private bool _hasCaps;

        [MenuItem("Window/NRR/Model Manager")]
        public static void Open()
        {
            var window = GetWindow<NRRModelManagerWindow>("NRR Model Manager");
            window.minSize = new Vector2(420, 420);
            window.Show();
        }

        private void OnEnable()
        {
            LoadSettings();
        }

        private void LoadSettings()
        {
            _settings = CreateInstance<NRRSettings>();
            _settings.PreferredBackend = EditorPrefs.GetString("NRR.PreferredBackend", "");
            _settings.DefaultModelPath = EditorPrefs.GetString("NRR.DefaultModelPath", "");
            _settings.DefaultFacialReferencePath = EditorPrefs.GetString("NRR.DefaultFacialReferencePath", "");
            _settings.InputWidth = EditorPrefs.GetInt("NRR.InputWidth", 640);
            _settings.InputHeight = EditorPrefs.GetInt("NRR.InputHeight", 360);
        }

        private void SaveSettings()
        {
            EditorPrefs.SetString("NRR.PreferredBackend", _settings.PreferredBackend ?? "");
            EditorPrefs.SetString("NRR.DefaultModelPath", _settings.DefaultModelPath ?? "");
            EditorPrefs.SetString("NRR.DefaultFacialReferencePath", _settings.DefaultFacialReferencePath ?? "");
            EditorPrefs.SetInt("NRR.InputWidth", _settings.InputWidth);
            EditorPrefs.SetInt("NRR.InputHeight", _settings.InputHeight);
        }

        private void OnGUI()
        {
            EditorGUILayout.LabelField("NRR Model Manager", EditorStyles.boldLabel);
            EditorGUILayout.Space();

            _settings.PreferredBackend = EditorGUILayout.TextField("Preferred Backend", _settings.PreferredBackend);
            _settings.DefaultModelPath = EditorGUILayout.TextField("Model Path (.nrrmodel)", _settings.DefaultModelPath);
            _settings.DefaultFacialReferencePath =
                EditorGUILayout.TextField("Facial Reference (.nrrref)", _settings.DefaultFacialReferencePath);
            EditorGUILayout.BeginHorizontal();
            _settings.InputWidth = EditorGUILayout.IntField("Input Width", _settings.InputWidth);
            _settings.InputHeight = EditorGUILayout.IntField("Height", _settings.InputHeight);
            EditorGUILayout.EndHorizontal();

            if (GUILayout.Button("Save Settings"))
            {
                SaveSettings();
                EditorUtility.SetDirty(_settings);
                Debug.Log("[NRR] Settings saved.");
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Device", EditorStyles.boldLabel);

            if (_device == null || !_device.IsValid)
            {
                if (GUILayout.Button("Create Device"))
                {
                    TryAction("create device", () =>
                    {
                        _device = NRRDevice.Create(new NRRDeviceOptions
                        {
                            preferred_backend = string.IsNullOrEmpty(_settings.PreferredBackend)
                                ? null : _settings.PreferredBackend,
                            frames_in_flight = 2,
                            enable_debugging = 1,
                            force_backend = 0,
                        });
                        _caps = _device.GetCapabilities();
                        _hasCaps = true;
                        _status = $"Device: {_caps.device_name} ({_caps.active_backend} v{_caps.backend_version})";
                    });
                }
            }
            else
            {
                EditorGUILayout.HelpBox(_status, MessageType.Info);
                if (GUILayout.Button("Destroy Device"))
                {
                    TryAction("destroy device", () =>
                    {
                        _device?.Dispose();
                        _device = null;
                        _model = null;
                        _reference = null;
                        _hasCaps = false;
                        _status = "Device destroyed.";
                    });
                }

                if (_hasCaps)
                {
                    DrawCapabilities(_caps);
                }
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Assets", EditorStyles.boldLabel);
            GUI.enabled = _device != null && _device.IsValid;

            if (GUILayout.Button("Load Model"))
            {
                TryAction("load model", () =>
                {
                    _model = _device.LoadModel(_settings.DefaultModelPath);
                    _status = $"Model loaded: {_settings.DefaultModelPath}\nInfo: {_model.GetInfo()}";
                });
            }

            if (GUILayout.Button("Load Facial Reference"))
            {
                TryAction("load reference", () =>
                {
                    _reference = _device.LoadReference(_settings.DefaultFacialReferencePath);
                    _status = $"Reference loaded: {_settings.DefaultFacialReferencePath}\n" +
                              $"ID: {_reference.GetId()}\nProvenance: {_reference.GetProvenance()}";
                });
            }

            if (_model != null && GUILayout.Button("Unload Model"))
            {
                _model.Dispose();
                _model = null;
                _status = "Model unloaded.";
            }

            if (_reference != null && GUILayout.Button("Unload Reference"))
            {
                _reference.Dispose();
                _reference = null;
                _status = "Reference unloaded.";
            }

            GUI.enabled = true;

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Status", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(_status, MessageType.None);
        }

        private void DrawCapabilities(NRRCapabilities caps)
        {
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Capabilities", EditorStyles.boldLabel);
            EditorGUI.indentLevel++;
            EditorGUILayout.LabelField("Vendor", caps.device_vendor);
            EditorGUILayout.LabelField("Type", caps.device_type);
            EditorGUILayout.LabelField("VRAM (MB)", caps.vram_mb.ToString());
            EditorGUILayout.LabelField("Neural Acceleration", caps.neural_acceleration.ToString());
            EditorGUILayout.LabelField("FP16", caps.fp16.ToString());
            EditorGUILayout.LabelField("Reference Conditioning", caps.reference_conditioning.ToString());
            EditorGUILayout.LabelField("Temporal Coherence", caps.temporal_coherence.ToString());
            EditorGUILayout.LabelField("Execution Score", caps.model_execution_score.ToString("F3"));
            EditorGUI.indentLevel--;
        }

        private void TryAction(string name, Action action)
        {
            try
            {
                action();
            }
            catch (NRRException e)
            {
                _status = $"Failed to {name}: {e.Message}";
                Debug.LogError(_status);
            }
            catch (Exception e)
            {
                _status = $"Failed to {name}: {e.Message}";
                Debug.LogError(_status);
            }
        }

        private void OnDisable()
        {
            _device?.Dispose();
            _device = null;
        }
    }
}