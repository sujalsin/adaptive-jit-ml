#include "jit_compiler.hpp"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Support/Error.h"
#include <chrono>

namespace adaptive_jit {

// MLOptimizer implementation
class JITCompiler::MLOptimizer {
public:
    MLOptimizer() = default;
    
    bool initialize(const std::string& model_path) {
        // TODO: Implement Python bridge for ML model
        return true;
    }
    
    std::vector<int> getOptimizationStrategy(const std::vector<double>& features) {
        // Placeholder: Return default optimization strategy
        return {3, 2, 1}; // Example optimization levels
    }
};

JITCompiler::JITCompiler()
    : Context(std::make_unique<llvm::LLVMContext>()),
      Builder(std::make_unique<llvm::IRBuilder<>>(*Context)),
      Optimizer(std::make_unique<MLOptimizer>()) {
    
    // Initialize LLJIT with default settings
    auto JITBuilder = llvm::orc::LLJITBuilder();
    auto JITOrErr = JITBuilder.create();
    
    if (auto Err = JITOrErr.takeError()) {
        llvm::errs() << "Failed to create LLJIT: " << Err << "\n";
        return;
    }
    
    JIT = std::move(*JITOrErr);
}

JITCompiler::~JITCompiler() = default;

bool JITCompiler::initialize(const std::string& ml_model_path) {
    return Optimizer->initialize(ml_model_path);
}

llvm::Expected<llvm::orc::ThreadSafeModule> JITCompiler::optimizeModule(
    std::unique_ptr<llvm::Module> M,
    const std::vector<double>& runtime_features) {
    
    // Collect additional runtime features
    collectRuntimeFeatures(*M);
    
    // Get optimization strategy from ML model
    auto strategy = Optimizer->getOptimizationStrategy(runtime_features);
    
    // Apply optimizations
    applyOptimizations(*M, strategy);
    
    // Create thread-safe module
    return llvm::orc::ThreadSafeModule(std::move(M), std::make_unique<llvm::LLVMContext>());
}

void JITCompiler::collectRuntimeFeatures(llvm::Module& M) {
    llvm::legacy::FunctionPassManager FPM(&M);
    llvm::legacy::PassManager MPM;
    
    // Add analysis passes
    FPM.add(llvm::createLoopInfoWrapperPass());
    FPM.add(llvm::createScalarEvolutionWrapperPass());
    
    // Initialize pass managers
    FPM.doInitialization();
    
    // Collect features for each function
    for (auto& F : M) {
        if (!F.isDeclaration()) {
            // Basic block count
            size_t bb_count = F.size();
            
            // Instruction count
            size_t inst_count = 0;
            for (const auto& BB : F) {
                inst_count += BB.size();
            }
            
            // Run function analysis passes
            FPM.run(F);
            
            // Store features in cache
            std::string key = F.getName().str();
            OptimizationCache[key] = {
                static_cast<int>(bb_count),
                static_cast<int>(inst_count)
            };
        }
    }
}

void JITCompiler::applyOptimizations(llvm::Module& M, const std::vector<int>& strategy) {
    llvm::legacy::PassManager PM;
    llvm::legacy::FunctionPassManager FPM(&M);
    
    // Create pass builder
    llvm::PassManagerBuilder PMBuilder;
    
    // Set optimization level based on strategy
    PMBuilder.OptLevel = strategy[0];
    PMBuilder.SizeLevel = strategy[1];
    
    // Add optimization passes based on strategy
    if (strategy[0] >= 2) {
        PMBuilder.Inliner = llvm::createFunctionInliningPass(strategy[0], strategy[1]);
        
        // Add aggressive optimization passes
        PM.add(llvm::createGlobalDCEPass());
        PM.add(llvm::createConstantMergePass());
        PM.add(llvm::createDeadStoreEliminationPass());
        
        if (strategy[0] >= 3) {
            PM.add(llvm::createAggressiveDCEPass());
            PM.add(llvm::createCFGSimplificationPass());
            PM.add(llvm::createInstructionCombiningPass());
        }
    }
    
    // Populate pass managers
    PMBuilder.populateModulePassManager(PM);
    PMBuilder.populateFunctionPassManager(FPM);
    
    // Run optimizations
    FPM.doInitialization();
    for (auto& F : M) {
        if (!F.isDeclaration()) {
            FPM.run(F);
        }
    }
    FPM.doFinalization();
    PM.run(M);
}

llvm::Error JITCompiler::addModule(llvm::orc::ThreadSafeModule TSM) {
    return JIT->addIRModule(std::move(TSM));
}

llvm::Expected<llvm::JITEagerSymbol> JITCompiler::lookup(const std::string& Name) {
    return JIT->lookup(Name);
}

} // namespace adaptive_jit
