//===-- PluginManager.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_PLUGINMANAGER_H
#define LLDB_CORE_PLUGINMANAGER_H

#include "lldb/Core/Architecture.h"
#include "lldb/Interpreter/Interfaces/ScriptedInterfaceUsages.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Target/Statistics.h"
#include "lldb/Utility/CompletionRequest.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-private-interfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#define LLDB_PLUGIN_DEFINE_ADV(ClassName, PluginName)                          \
  namespace lldb_private {                                                     \
  void lldb_initialize_##PluginName() { ClassName::Initialize(); }             \
  void lldb_terminate_##PluginName() { ClassName::Terminate(); }               \
  }

#define LLDB_PLUGIN_DEFINE(PluginName)                                         \
  LLDB_PLUGIN_DEFINE_ADV(PluginName, PluginName)

// FIXME: Generate me with CMake
#define LLDB_PLUGIN_DECLARE(PluginName)                                        \
  namespace lldb_private {                                                     \
  extern void lldb_initialize_##PluginName();                                  \
  extern void lldb_terminate_##PluginName();                                   \
  }

#define LLDB_PLUGIN_INITIALIZE(PluginName) lldb_initialize_##PluginName()
#define LLDB_PLUGIN_TERMINATE(PluginName) lldb_terminate_##PluginName()

namespace lldb_private {
class CommandInterpreter;
class Debugger;
class StringList;

struct RegisteredPluginInfo {
  llvm::StringRef name = "";
  llvm::StringRef description = "";
  bool enabled = false;
};

// Define some data structures to describe known plugin "namespaces".
// The PluginManager is organized into a series of static functions
// that operate on different types of plugins. For example SystemRuntime
// and ObjectFile plugins.
//
// The namespace name is used a prefix when matching plugin names. For example,
// if we have an "macosx" plugin in the "system-runtime" namespace then we will
// match a plugin name pattern against the "system-runtime.macosx" name.
//
// The plugin namespace here is used so we can operate on all the plugins
// of a given type so it is easy to enable or disable them as a group.
using GetPluginInfo = std::function<std::vector<RegisteredPluginInfo>()>;
using SetPluginEnabled = std::function<bool(llvm::StringRef, bool)>;
struct PluginNamespace {
  llvm::StringRef name;
  GetPluginInfo get_info;
  SetPluginEnabled set_enabled;
};

class PluginManager {
public:
  static void Initialize();

  static void Terminate();

  // Support for enabling and disabling plugins.

  // Return the plugins that can be enabled or disabled by the user.
  static llvm::ArrayRef<PluginNamespace> GetPluginNamespaces();

  // Generate a json object that describes the plugins that are available.
  // This is a json representation of the plugin info returned by
  // GetPluginNamespaces().
  //
  //    {
  //       <plugin-namespace>: [
  //           {
  //               "enabled": <bool>,
  //               "name": <plugin-name>,
  //           },
  //           ...
  //       ],
  //       ...
  //    }
  //
  // If pattern is given it will be used to filter the plugins that are
  // are returned. The pattern filters the plugin names using the
  // PluginManager::MatchPluginName() function.
  static llvm::json::Object GetJSON(llvm::StringRef pattern = "");

  // Return true if the pattern matches the plugin name.
  //
  // The pattern matches the name if it is exactly equal to the namespace name
  // or if it is equal to the qualified name, which is the namespace name
  // followed by a dot and the plugin name (e.g. "system-runtime.foo").
  //
  // An empty pattern matches all plugins.
  static bool MatchPluginName(llvm::StringRef pattern,
                              const PluginNamespace &plugin_ns,
                              const RegisteredPluginInfo &plugin);

  // ABI
  static bool RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                             ABICreateInstance create_callback);

  static bool UnregisterPlugin(ABICreateInstance create_callback);

  static ABICreateInstance GetABICreateCallbackAtIndex(uint32_t idx);

  // Architecture
  static void RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                             ArchitectureCreateInstance create_callback);

  static void UnregisterPlugin(ArchitectureCreateInstance create_callback);

  static std::unique_ptr<Architecture>
  CreateArchitectureInstance(const ArchSpec &arch);

  // Disassembler
  static bool RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                             DisassemblerCreateInstance create_callback);

  static bool UnregisterPlugin(DisassemblerCreateInstance create_callback);

  static DisassemblerCreateInstance
  GetDisassemblerCreateCallbackAtIndex(uint32_t idx);

  static DisassemblerCreateInstance
  GetDisassemblerCreateCallbackForPluginName(llvm::StringRef name);

  // DynamicLoader
  static bool
  RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                 DynamicLoaderCreateInstance create_callback,
                 DebuggerInitializeCallback debugger_init_callback = nullptr);

  static bool UnregisterPlugin(DynamicLoaderCreateInstance create_callback);

  static DynamicLoaderCreateInstance
  GetDynamicLoaderCreateCallbackAtIndex(uint32_t idx);

  static DynamicLoaderCreateInstance
  GetDynamicLoaderCreateCallbackForPluginName(llvm::StringRef name);

  // JITLoader
  static bool
  RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                 JITLoaderCreateInstance create_callback,
                 DebuggerInitializeCallback debugger_init_callback = nullptr);

  static bool UnregisterPlugin(JITLoaderCreateInstance create_callback);

  static JITLoaderCreateInstance
  GetJITLoaderCreateCallbackAtIndex(uint32_t idx);

  // EmulateInstruction
  static bool RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                             EmulateInstructionCreateInstance create_callback);

  static bool
  UnregisterPlugin(EmulateInstructionCreateInstance create_callback);

  static EmulateInstructionCreateInstance
  GetEmulateInstructionCreateCallbackAtIndex(uint32_t idx);

  static EmulateInstructionCreateInstance
  GetEmulateInstructionCreateCallbackForPluginName(llvm::StringRef name);

  // OperatingSystem
  static bool RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                             OperatingSystemCreateInstance create_callback,
                             DebuggerInitializeCallback debugger_init_callback);

  static bool UnregisterPlugin(OperatingSystemCreateInstance create_callback);

  static OperatingSystemCreateInstance
  GetOperatingSystemCreateCallbackAtIndex(uint32_t idx);

  static OperatingSystemCreateInstance
  GetOperatingSystemCreateCallbackForPluginName(llvm::StringRef name);

  // Language
  static bool
  RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                 LanguageCreateInstance create_callback,
                 DebuggerInitializeCallback debugger_init_callback = nullptr);

  static bool UnregisterPlugin(LanguageCreateInstance create_callback);

  static LanguageCreateInstance GetLanguageCreateCallbackAtIndex(uint32_t idx);

  // LanguageRuntime
  static bool RegisterPlugin(
      llvm::StringRef name, llvm::StringRef description,
      LanguageRuntimeCreateInstance create_callback,
      LanguageRuntimeGetCommandObject command_callback = nullptr,
      LanguageRuntimeGetExceptionPrecondition precondition_callback = nullptr);

  static bool UnregisterPlugin(LanguageRuntimeCreateInstance create_callback);

  static LanguageRuntimeCreateInstance
  GetLanguageRuntimeCreateCallbackAtIndex(uint32_t idx);

  static LanguageRuntimeGetCommandObject
  GetLanguageRuntimeGetCommandObjectAtIndex(uint32_t idx);

  static LanguageRuntimeGetExceptionPrecondition
  GetLanguageRuntimeGetExceptionPreconditionAtIndex(uint32_t idx);

  // SystemRuntime
  static bool RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                             SystemRuntimeCreateInstance create_callback);

  static bool UnregisterPlugin(SystemRuntimeCreateInstance create_callback);

  static SystemRuntimeCreateInstance
  GetSystemRuntimeCreateCallbackAtIndex(uint32_t idx);

  // ObjectFile
  static bool
  RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                 ObjectFileCreateInstance create_callback,
                 ObjectFileCreateMemoryInstance create_memory_callback,
                 ObjectFileGetModuleSpecifications get_module_specifications,
                 ObjectFileSaveCore save_core = nullptr,
                 DebuggerInitializeCallback debugger_init_callback = nullptr);

  static bool UnregisterPlugin(ObjectFileCreateInstance create_callback);

  static bool IsRegisteredObjectFilePluginName(llvm::StringRef name);

  static ObjectFileCreateInstance
  GetObjectFileCreateCallbackAtIndex(uint32_t idx);

  static ObjectFileCreateMemoryInstance
  GetObjectFileCreateMemoryCallbackAtIndex(uint32_t idx);

  static ObjectFileGetModuleSpecifications
  GetObjectFileGetModuleSpecificationsCallbackAtIndex(uint32_t idx);

  static ObjectFileCreateMemoryInstance
  GetObjectFileCreateMemoryCallbackForPluginName(llvm::StringRef name);

  static Status SaveCore(lldb_private::SaveCoreOptions &core_options);

  static std::vector<llvm::StringRef> GetSaveCorePluginNames();

  // ObjectContainer
  static bool RegisterPlugin(
      llvm::StringRef name, llvm::StringRef description,
      ObjectContainerCreateInstance create_callback,
      ObjectFileGetModuleSpecifications get_module_specifications,
      ObjectContainerCreateMemoryInstance create_memory_callback = nullptr);

  static bool UnregisterPlugin(ObjectContainerCreateInstance create_callback);

  static ObjectContainerCreateInstance
  GetObjectContainerCreateCallbackAtIndex(uint32_t idx);

  static ObjectContainerCreateMemoryInstance
  GetObjectContainerCreateMemoryCallbackAtIndex(uint32_t idx);

  static ObjectFileGetModuleSpecifications
  GetObjectContainerGetModuleSpecificationsCallbackAtIndex(uint32_t idx);

  // Platform
  static bool
  RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                 PlatformCreateInstance create_callback,
                 DebuggerInitializeCallback debugger_init_callback = nullptr);

  static bool UnregisterPlugin(PlatformCreateInstance create_callback);

  static PlatformCreateInstance GetPlatformCreateCallbackAtIndex(uint32_t idx);

  static PlatformCreateInstance
  GetPlatformCreateCallbackForPluginName(llvm::StringRef name);

  static llvm::StringRef GetPlatformPluginNameAtIndex(uint32_t idx);

  static llvm::StringRef GetPlatformPluginDescriptionAtIndex(uint32_t idx);

  static void AutoCompletePlatformName(llvm::StringRef partial_name,
                                       CompletionRequest &request);
  // Process
  static bool
  RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                 ProcessCreateInstance create_callback,
                 DebuggerInitializeCallback debugger_init_callback = nullptr);

  static bool UnregisterPlugin(ProcessCreateInstance create_callback);

  static ProcessCreateInstance GetProcessCreateCallbackAtIndex(uint32_t idx);

  static ProcessCreateInstance
  GetProcessCreateCallbackForPluginName(llvm::StringRef name);

  static llvm::StringRef GetProcessPluginNameAtIndex(uint32_t idx);

  static llvm::StringRef GetProcessPluginDescriptionAtIndex(uint32_t idx);

  static void AutoCompleteProcessName(llvm::StringRef partial_name,
                                      CompletionRequest &request);

  // Protocol
  static bool RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                             ProtocolServerCreateInstance create_callback);

  static bool UnregisterPlugin(ProtocolServerCreateInstance create_callback);

  static llvm::StringRef GetProtocolServerPluginNameAtIndex(uint32_t idx);

  static ProtocolServerCreateInstance
  GetProtocolCreateCallbackForPluginName(llvm::StringRef name);

  // Register Type Provider
  static bool RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                             RegisterTypeBuilderCreateInstance create_callback);

  static bool
  UnregisterPlugin(RegisterTypeBuilderCreateInstance create_callback);

  static lldb::RegisterTypeBuilderSP GetRegisterTypeBuilder(Target &target);

  // ScriptInterpreter
  static bool RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                             lldb::ScriptLanguage script_lang,
                             ScriptInterpreterCreateInstance create_callback);

  static bool UnregisterPlugin(ScriptInterpreterCreateInstance create_callback);

  static ScriptInterpreterCreateInstance
  GetScriptInterpreterCreateCallbackAtIndex(uint32_t idx);

  static lldb::ScriptInterpreterSP
  GetScriptInterpreterForLanguage(lldb::ScriptLanguage script_lang,
                                  Debugger &debugger);

  // StructuredDataPlugin

  /// Register a StructuredDataPlugin class along with optional
  /// callbacks for debugger initialization and Process launch info
  /// filtering and manipulation.
  ///
  /// \param[in] name
  ///    The name of the plugin.
  ///
  /// \param[in] description
  ///    A description string for the plugin.
  ///
  /// \param[in] create_callback
  ///    The callback that will be invoked to create an instance of
  ///    the callback.  This may not be nullptr.
  ///
  /// \param[in] debugger_init_callback
  ///    An optional callback that will be made when a Debugger
  ///    instance is initialized.
  ///
  /// \param[in] filter_callback
  ///    An optional callback that will be invoked before LLDB
  ///    launches a process for debugging.  The callback must
  ///    do the following:
  ///    1. Only do something if the plugin's behavior is enabled.
  ///    2. Only make changes for processes that are relevant to the
  ///       plugin.  The callback gets a pointer to the Target, which
  ///       can be inspected as needed.  The ProcessLaunchInfo is
  ///       provided in read-write mode, and may be modified by the
  ///       plugin if, for instance, additional environment variables
  ///       are needed to support the feature when enabled.
  ///
  /// \return
  ///    Returns true upon success; otherwise, false.
  static bool
  RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                 StructuredDataPluginCreateInstance create_callback,
                 DebuggerInitializeCallback debugger_init_callback = nullptr,
                 StructuredDataFilterLaunchInfo filter_callback = nullptr);

  static bool
  UnregisterPlugin(StructuredDataPluginCreateInstance create_callback);

  static StructuredDataPluginCreateInstance
  GetStructuredDataPluginCreateCallbackAtIndex(uint32_t idx);

  static StructuredDataFilterLaunchInfo
  GetStructuredDataFilterCallbackAtIndex(uint32_t idx,
                                         bool &iteration_complete);

  // SymbolFile
  static bool
  RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                 SymbolFileCreateInstance create_callback,
                 DebuggerInitializeCallback debugger_init_callback = nullptr);

  static bool UnregisterPlugin(SymbolFileCreateInstance create_callback);

  static SymbolFileCreateInstance
  GetSymbolFileCreateCallbackAtIndex(uint32_t idx);

  // SymbolVendor
  static bool RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                             SymbolVendorCreateInstance create_callback);

  static bool UnregisterPlugin(SymbolVendorCreateInstance create_callback);

  static SymbolVendorCreateInstance
  GetSymbolVendorCreateCallbackAtIndex(uint32_t idx);

  // SymbolLocator
  static bool RegisterPlugin(
      llvm::StringRef name, llvm::StringRef description,
      SymbolLocatorCreateInstance create_callback,
      SymbolLocatorLocateExecutableObjectFile locate_executable_object_file =
          nullptr,
      SymbolLocatorLocateExecutableSymbolFile locate_executable_symbol_file =
          nullptr,
      SymbolLocatorDownloadObjectAndSymbolFile download_object_symbol_file =
          nullptr,
      SymbolLocatorFindSymbolFileInBundle find_symbol_file_in_bundle = nullptr,
      DebuggerInitializeCallback debugger_init_callback = nullptr);

  static bool UnregisterPlugin(SymbolLocatorCreateInstance create_callback);

  static SymbolLocatorCreateInstance
  GetSymbolLocatorCreateCallbackAtIndex(uint32_t idx);

  static ModuleSpec LocateExecutableObjectFile(const ModuleSpec &module_spec,
                                               StatisticsMap &map);

  static FileSpec
  LocateExecutableSymbolFile(const ModuleSpec &module_spec,
                             const FileSpecList &default_search_paths,
                             StatisticsMap &map);

  static bool DownloadObjectAndSymbolFile(ModuleSpec &module_spec,
                                          Status &error,
                                          bool force_lookup = true,
                                          bool copy_executable = true);

  static FileSpec FindSymbolFileInBundle(const FileSpec &dsym_bundle_fspec,
                                         const UUID *uuid,
                                         const ArchSpec *arch);

  // Trace
  static bool RegisterPlugin(
      llvm::StringRef name, llvm::StringRef description,
      TraceCreateInstanceFromBundle create_callback_from_bundle,
      TraceCreateInstanceForLiveProcess create_callback_for_live_process,
      llvm::StringRef schema,
      DebuggerInitializeCallback debugger_init_callback);

  static bool
  UnregisterPlugin(TraceCreateInstanceFromBundle create_callback);

  static TraceCreateInstanceFromBundle
  GetTraceCreateCallback(llvm::StringRef plugin_name);

  static TraceCreateInstanceForLiveProcess
  GetTraceCreateCallbackForLiveProcess(llvm::StringRef plugin_name);

  /// Get the JSON schema for a trace bundle description file corresponding to
  /// the given plugin.
  ///
  /// \param[in] plugin_name
  ///     The name of the plugin.
  ///
  /// \return
  ///     An empty \a StringRef if no plugin was found with that plugin name,
  ///     otherwise the actual schema is returned.
  static llvm::StringRef GetTraceSchema(llvm::StringRef plugin_name);

  /// Get the JSON schema for a trace bundle description file corresponding to
  /// the plugin given by its index.
  ///
  /// \param[in] index
  ///     The index of the plugin to get the schema of.
  ///
  /// \return
  ///     An empty \a StringRef if the index is greater than or equal to the
  ///     number plugins, otherwise the actual schema is returned.
  static llvm::StringRef GetTraceSchema(size_t index);

  // TraceExporter

  /// \param[in] create_thread_trace_export_command
  ///     This callback is used to create a CommandObject that will be listed
  ///     under "thread trace export". Can be \b null.
  static bool RegisterPlugin(
      llvm::StringRef name, llvm::StringRef description,
      TraceExporterCreateInstance create_callback,
      ThreadTraceExportCommandCreator create_thread_trace_export_command);

  static TraceExporterCreateInstance
  GetTraceExporterCreateCallback(llvm::StringRef plugin_name);

  static bool UnregisterPlugin(TraceExporterCreateInstance create_callback);

  static llvm::StringRef GetTraceExporterPluginNameAtIndex(uint32_t index);

  /// Return the callback used to create the CommandObject that will be listed
  /// under "thread trace export". Can be \b null.
  static ThreadTraceExportCommandCreator
  GetThreadTraceExportCommandCreatorAtIndex(uint32_t index);

  // UnwindAssembly
  static bool RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                             UnwindAssemblyCreateInstance create_callback);

  static bool UnregisterPlugin(UnwindAssemblyCreateInstance create_callback);

  static UnwindAssemblyCreateInstance
  GetUnwindAssemblyCreateCallbackAtIndex(uint32_t idx);

  // MemoryHistory
  static bool RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                             MemoryHistoryCreateInstance create_callback);

  static bool UnregisterPlugin(MemoryHistoryCreateInstance create_callback);

  static MemoryHistoryCreateInstance
  GetMemoryHistoryCreateCallbackAtIndex(uint32_t idx);

  // InstrumentationRuntime
  static bool
  RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                 InstrumentationRuntimeCreateInstance create_callback,
                 InstrumentationRuntimeGetType get_type_callback);

  static bool
  UnregisterPlugin(InstrumentationRuntimeCreateInstance create_callback);

  static InstrumentationRuntimeGetType
  GetInstrumentationRuntimeGetTypeCallbackAtIndex(uint32_t idx);

  static InstrumentationRuntimeCreateInstance
  GetInstrumentationRuntimeCreateCallbackAtIndex(uint32_t idx);

  // TypeSystem
  static bool RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                             TypeSystemCreateInstance create_callback,
                             LanguageSet supported_languages_for_types,
                             LanguageSet supported_languages_for_expressions);

  static bool UnregisterPlugin(TypeSystemCreateInstance create_callback);

  static TypeSystemCreateInstance
  GetTypeSystemCreateCallbackAtIndex(uint32_t idx);

  static LanguageSet GetAllTypeSystemSupportedLanguagesForTypes();

  static LanguageSet GetAllTypeSystemSupportedLanguagesForExpressions();

  // Scripted Interface
  static bool RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                             ScriptedInterfaceCreateInstance create_callback,
                             lldb::ScriptLanguage language,
                             ScriptedInterfaceUsages usages);

  static bool UnregisterPlugin(ScriptedInterfaceCreateInstance create_callback);

  static uint32_t GetNumScriptedInterfaces();

  static llvm::StringRef GetScriptedInterfaceNameAtIndex(uint32_t idx);

  static llvm::StringRef GetScriptedInterfaceDescriptionAtIndex(uint32_t idx);

  static lldb::ScriptLanguage GetScriptedInterfaceLanguageAtIndex(uint32_t idx);

  static ScriptedInterfaceUsages
  GetScriptedInterfaceUsagesAtIndex(uint32_t idx);

  // REPL
  static bool RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                             REPLCreateInstance create_callback,
                             LanguageSet supported_languages);

  static bool UnregisterPlugin(REPLCreateInstance create_callback);

  static REPLCreateInstance GetREPLCreateCallbackAtIndex(uint32_t idx);

  static LanguageSet GetREPLSupportedLanguagesAtIndex(uint32_t idx);

  static LanguageSet GetREPLAllTypeSystemSupportedLanguages();

  // Some plug-ins might register a DebuggerInitializeCallback callback when
  // registering the plug-in. After a new Debugger instance is created, this
  // DebuggerInitialize function will get called. This allows plug-ins to
  // install Properties and do any other initialization that requires a
  // debugger instance.
  static void DebuggerInitialize(Debugger &debugger);

  static lldb::OptionValuePropertiesSP
  GetSettingForDynamicLoaderPlugin(Debugger &debugger,
                                   llvm::StringRef setting_name);

  static bool CreateSettingForDynamicLoaderPlugin(
      Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
      llvm::StringRef description, bool is_global_property);

  static lldb::OptionValuePropertiesSP
  GetSettingForPlatformPlugin(Debugger &debugger, llvm::StringRef setting_name);

  static bool CreateSettingForPlatformPlugin(
      Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
      llvm::StringRef description, bool is_global_property);

  static lldb::OptionValuePropertiesSP
  GetSettingForProcessPlugin(Debugger &debugger, llvm::StringRef setting_name);

  static bool CreateSettingForProcessPlugin(
      Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
      llvm::StringRef description, bool is_global_property);

  static lldb::OptionValuePropertiesSP
  GetSettingForSymbolLocatorPlugin(Debugger &debugger,
                                   llvm::StringRef setting_name);

  static bool CreateSettingForSymbolLocatorPlugin(
      Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
      llvm::StringRef description, bool is_global_property);

  static bool CreateSettingForTracePlugin(
      Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
      llvm::StringRef description, bool is_global_property);

  static lldb::OptionValuePropertiesSP
  GetSettingForObjectFilePlugin(Debugger &debugger,
                                llvm::StringRef setting_name);

  static bool CreateSettingForObjectFilePlugin(
      Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
      llvm::StringRef description, bool is_global_property);

  static lldb::OptionValuePropertiesSP
  GetSettingForSymbolFilePlugin(Debugger &debugger,
                                llvm::StringRef setting_name);

  static bool CreateSettingForSymbolFilePlugin(
      Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
      llvm::StringRef description, bool is_global_property);

  static lldb::OptionValuePropertiesSP
  GetSettingForJITLoaderPlugin(Debugger &debugger,
                               llvm::StringRef setting_name);

  static bool CreateSettingForJITLoaderPlugin(
      Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
      llvm::StringRef description, bool is_global_property);

  static lldb::OptionValuePropertiesSP
  GetSettingForOperatingSystemPlugin(Debugger &debugger,
                                     llvm::StringRef setting_name);

  static bool CreateSettingForOperatingSystemPlugin(
      Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
      llvm::StringRef description, bool is_global_property);

  static lldb::OptionValuePropertiesSP
  GetSettingForStructuredDataPlugin(Debugger &debugger,
                                    llvm::StringRef setting_name);

  static bool CreateSettingForStructuredDataPlugin(
      Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
      llvm::StringRef description, bool is_global_property);

  static lldb::OptionValuePropertiesSP
  GetSettingForCPlusPlusLanguagePlugin(Debugger &debugger,
                                       llvm::StringRef setting_name);

  static bool CreateSettingForCPlusPlusLanguagePlugin(
      Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
      llvm::StringRef description, bool is_global_property);

  //
  // Plugin Info+Enable Declarations
  //
  static std::vector<RegisteredPluginInfo> GetABIPluginInfo();
  static bool SetABIPluginEnabled(llvm::StringRef name, bool enable);

  static std::vector<RegisteredPluginInfo> GetArchitecturePluginInfo();
  static bool SetArchitecturePluginEnabled(llvm::StringRef name, bool enable);

  static std::vector<RegisteredPluginInfo> GetDisassemblerPluginInfo();
  static bool SetDisassemblerPluginEnabled(llvm::StringRef name, bool enable);

  static std::vector<RegisteredPluginInfo> GetDynamicLoaderPluginInfo();
  static bool SetDynamicLoaderPluginEnabled(llvm::StringRef name, bool enable);

  static std::vector<RegisteredPluginInfo> GetEmulateInstructionPluginInfo();
  static bool SetEmulateInstructionPluginEnabled(llvm::StringRef name,
                                                 bool enable);

  static std::vector<RegisteredPluginInfo>
  GetInstrumentationRuntimePluginInfo();
  static bool SetInstrumentationRuntimePluginEnabled(llvm::StringRef name,
                                                     bool enable);

  static std::vector<RegisteredPluginInfo> GetJITLoaderPluginInfo();
  static bool SetJITLoaderPluginEnabled(llvm::StringRef name, bool enable);

  static std::vector<RegisteredPluginInfo> GetLanguagePluginInfo();
  static bool SetLanguagePluginEnabled(llvm::StringRef name, bool enable);

  static std::vector<RegisteredPluginInfo> GetLanguageRuntimePluginInfo();
  static bool SetLanguageRuntimePluginEnabled(llvm::StringRef name,
                                              bool enable);

  static std::vector<RegisteredPluginInfo> GetMemoryHistoryPluginInfo();
  static bool SetMemoryHistoryPluginEnabled(llvm::StringRef name, bool enable);

  static std::vector<RegisteredPluginInfo> GetObjectContainerPluginInfo();
  static bool SetObjectContainerPluginEnabled(llvm::StringRef name,
                                              bool enable);

  static std::vector<RegisteredPluginInfo> GetObjectFilePluginInfo();
  static bool SetObjectFilePluginEnabled(llvm::StringRef name, bool enable);

  static std::vector<RegisteredPluginInfo> GetOperatingSystemPluginInfo();
  static bool SetOperatingSystemPluginEnabled(llvm::StringRef name,
                                              bool enable);

  static std::vector<RegisteredPluginInfo> GetPlatformPluginInfo();
  static bool SetPlatformPluginEnabled(llvm::StringRef name, bool enable);

  static std::vector<RegisteredPluginInfo> GetProcessPluginInfo();
  static bool SetProcessPluginEnabled(llvm::StringRef name, bool enable);

  static std::vector<RegisteredPluginInfo> GetREPLPluginInfo();
  static bool SetREPLPluginEnabled(llvm::StringRef name, bool enable);

  static std::vector<RegisteredPluginInfo> GetRegisterTypeBuilderPluginInfo();
  static bool SetRegisterTypeBuilderPluginEnabled(llvm::StringRef name,
                                                  bool enable);

  static std::vector<RegisteredPluginInfo> GetScriptInterpreterPluginInfo();
  static bool SetScriptInterpreterPluginEnabled(llvm::StringRef name,
                                                bool enable);

  static std::vector<RegisteredPluginInfo> GetScriptedInterfacePluginInfo();
  static bool SetScriptedInterfacePluginEnabled(llvm::StringRef name,
                                                bool enable);

  static std::vector<RegisteredPluginInfo> GetStructuredDataPluginInfo();
  static bool SetStructuredDataPluginEnabled(llvm::StringRef name, bool enable);

  static std::vector<RegisteredPluginInfo> GetSymbolFilePluginInfo();
  static bool SetSymbolFilePluginEnabled(llvm::StringRef name, bool enable);

  static std::vector<RegisteredPluginInfo> GetSymbolLocatorPluginInfo();
  static bool SetSymbolLocatorPluginEnabled(llvm::StringRef name, bool enable);

  static std::vector<RegisteredPluginInfo> GetSymbolVendorPluginInfo();
  static bool SetSymbolVendorPluginEnabled(llvm::StringRef name, bool enable);

  static std::vector<RegisteredPluginInfo> GetSystemRuntimePluginInfo();
  static bool SetSystemRuntimePluginEnabled(llvm::StringRef name, bool enable);

  static std::vector<RegisteredPluginInfo> GetTracePluginInfo();
  static bool SetTracePluginEnabled(llvm::StringRef name, bool enable);

  static std::vector<RegisteredPluginInfo> GetTraceExporterPluginInfo();
  static bool SetTraceExporterPluginEnabled(llvm::StringRef name, bool enable);

  static std::vector<RegisteredPluginInfo> GetTypeSystemPluginInfo();
  static bool SetTypeSystemPluginEnabled(llvm::StringRef name, bool enable);

  static std::vector<RegisteredPluginInfo> GetUnwindAssemblyPluginInfo();
  static bool SetUnwindAssemblyPluginEnabled(llvm::StringRef name, bool enable);

  static void AutoCompletePluginName(llvm::StringRef partial_name,
                                     CompletionRequest &request);
};

} // namespace lldb_private

#endif // LLDB_CORE_PLUGINMANAGER_H
