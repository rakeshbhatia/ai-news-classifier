#!/usr/bin/env python3
"""
Test script for AI World Press classifier functions.
Tests classification accuracy for AI relevance, topic categorization, and key player extraction.
"""

import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import spacy
import tabulate
from termcolor import colored

# Import classification functions
from main import is_ai_related, determine_category, extract_key_player, nlp

# Test cases
TEST_CASES = [
    # AI-related news articles - Business & Finance
    {
        "text": "OpenAI announced today that it has secured $2 billion in funding from Microsoft and other investors, valuing the company at $20 billion. The investment will help OpenAI scale its operations and accelerate research into artificial general intelligence.",
        "is_ai_related": True,
        "expected_category": "Business & Finance",
        "expected_key_player": "OpenAI"
    },
    {
        "text": "Anthropic has raised $450 million in a Series C funding round led by Google. The AI safety and research company is now valued at over $5 billion as investors continue to pour money into generative AI startups.",
        "is_ai_related": True,
        "expected_category": "Business & Finance",
        "expected_key_player": "Anthropic"
    },
    {
        "text": "NVIDIA reported record quarterly earnings, with revenue up 200% year-over-year, driven by unprecedented demand for AI chips. The company's stock surged 12% on the news, pushing its market cap past $1 trillion.",
        "is_ai_related": True,
        "expected_category": "Business & Finance",
        "expected_key_player": "NVIDIA"
    },
    
    # AI-related news articles - Policy & Regulation
    {
        "text": "The European Union has passed the AI Act, the world's first comprehensive legal framework for artificial intelligence. The legislation categorizes AI systems based on risk levels and imposes strict requirements on high-risk applications.",
        "is_ai_related": True,
        "expected_category": "Policy & Regulation",
        "expected_key_player": "European Union"
    },
    {
        "text": "President Biden signed an executive order on AI safety, requiring companies to share safety test results with the government before releasing powerful AI models. The order also mandates watermarking of AI-generated content.",
        "is_ai_related": True,
        "expected_category": "Policy & Regulation",
        "expected_key_player": "Biden"
    },
    {
        "text": "China's Cyberspace Administration has introduced new regulations requiring all generative AI products to undergo security assessments before public release. The rules aim to prevent the generation of illegal content and ensure alignment with socialist values.",
        "is_ai_related": True,
        "expected_category": "Policy & Regulation",
        "expected_key_player": "China's Cyberspace Administration"
    },
    
    # AI-related news articles - Research & Development
    {
        "text": "Researchers at DeepMind have published a breakthrough paper in Nature, demonstrating an AI system that can predict protein folding with near-perfect accuracy. The AlphaFold 3 model represents a significant advance over previous versions.",
        "is_ai_related": True,
        "expected_category": "Research & Development",
        "expected_key_player": "DeepMind"
    },
    {
        "text": "A team of MIT computer scientists has developed a new technique for training large language models that reduces computational requirements by 50% while maintaining performance. The method could democratize access to advanced AI research.",
        "is_ai_related": True,
        "expected_category": "Research & Development",
        "expected_key_player": "MIT"
    },
    {
        "text": "Google Research unveiled a multimodal AI system capable of understanding and generating content across text, images, audio, and video from a single prompt. The model achieves state-of-the-art results on 20 different benchmarks.",
        "is_ai_related": True,
        "expected_category": "Research & Development",
        "expected_key_player": "Google Research"
    },
    
    # AI-related news articles - Ethics & Social Impact
    {
        "text": "A new study from Stanford's Institute for Human-Centered AI found significant racial and gender bias in widely used facial recognition systems. Researchers call for stronger oversight and testing requirements before deployment in sensitive applications.",
        "is_ai_related": True,
        "expected_category": "Ethics & Social Impact",
        "expected_key_player": "Stanford's Institute for Human-Centered AI"
    },
    {
        "text": "Amazon's AI-powered hiring tool was scrapped after it was discovered to be discriminating against female applicants. Internal documents reveal the system was trained primarily on resumes from male candidates, perpetuating existing gender imbalances.",
        "is_ai_related": True,
        "expected_category": "Ethics & Social Impact",
        "expected_key_player": "Amazon"
    },
    {
        "text": "A coalition of civil rights organizations has launched a campaign against the use of AI in welfare benefit determinations. The group argues that automated systems lack transparency and have led to wrongful benefit denials affecting vulnerable populations.",
        "is_ai_related": True,
        "expected_category": "Ethics & Social Impact",
        "expected_key_player": "civil rights organizations"
    },
    
    # AI-related news articles - Defense & Security
    {
        "text": "The Pentagon has awarded a $10 billion contract to develop AI-powered autonomous drones for surveillance and reconnaissance missions. Officials emphasized that human operators will maintain control over any lethal force decisions.",
        "is_ai_related": True,
        "expected_category": "Defense & Security",
        "expected_key_player": "Pentagon"
    },
    {
        "text": "NATO announced the establishment of a new AI Security Center to coordinate defense against AI-enabled cyber threats. The alliance will develop shared standards for securing military AI systems and countering disinformation campaigns.",
        "is_ai_related": True,
        "expected_category": "Defense & Security",
        "expected_key_player": "NATO"
    },
    {
        "text": "Russian hackers deployed advanced AI tools to breach government networks in three European countries, according to a new intelligence report. The attack used machine learning algorithms to evade detection and adapt to security responses.",
        "is_ai_related": True,
        "expected_category": "Defense & Security",
        "expected_key_player": "Russian hackers"
    },
    
    # AI-related news articles - Technology & Innovation
    {
        "text": "Apple unveiled its new M3 Ultra chip with dedicated neural processing units designed specifically for on-device AI applications. The company claims it's 4x faster at machine learning tasks than its predecessor while using 30% less power.",
        "is_ai_related": True,
        "expected_category": "Technology & Innovation",
        "expected_key_player": "Apple"
    },
    {
        "text": "Tesla's Full Self-Driving software has received a major update incorporating a completely redesigned neural network architecture. The company claims a 40% reduction in disengagements based on real-world testing data.",
        "is_ai_related": True,
        "expected_category": "Technology & Innovation",
        "expected_key_player": "Tesla"
    },
    {
        "text": "Microsoft has integrated AI capabilities across its entire Office suite, enabling automatic document summarization, smart email composition, and real-time translation in Teams meetings. The features leverage the company's latest Azure AI models.",
        "is_ai_related": True,
        "expected_category": "Technology & Innovation",
        "expected_key_player": "Microsoft"
    },
    
    # AI-related news articles - Healthcare & Biotech
    {
        "text": "Researchers at Mayo Clinic have developed an AI system that can detect early signs of pancreatic cancer from routine CT scans with 95% accuracy. The algorithm identified subtle patterns that are often missed by radiologists.",
        "is_ai_related": True,
        "expected_category": "Healthcare & Biotech",
        "expected_key_player": "Mayo Clinic"
    },
    {
        "text": "The FDA has approved the first AI-powered diagnostic tool for detecting diabetic retinopathy without physician oversight. The system analyzes retinal images and can be used in primary care settings, potentially increasing screening rates.",
        "is_ai_related": True,
        "expected_category": "Healthcare & Biotech",
        "expected_key_player": "FDA"
    },
    {
        "text": "Pfizer announced a partnership with Recursion Pharmaceuticals to use AI for drug discovery, committing $50 million upfront with potential milestone payments of up to $1.5 billion. The collaboration will focus on identifying novel targets for fibrotic diseases.",
        "is_ai_related": True,
        "expected_category": "Healthcare & Biotech",
        "expected_key_player": "Pfizer"
    },
    
    # AI-related news articles - Entertainment & Media
    {
        "text": "Netflix has implemented a new AI-powered content recommendation system that increases viewer engagement by 35%. The algorithm analyzes not just viewing history but also micro-interactions like pauses and rewinds.",
        "is_ai_related": True,
        "expected_category": "Entertainment & Media",
        "expected_key_player": "Netflix"
    },
    {
        "text": "Warner Bros. has faced criticism for using AI to generate background characters in its latest blockbuster, reducing jobs for extras. The Screen Actors Guild has called for industry-wide regulations on AI use in film production.",
        "is_ai_related": True,
        "expected_category": "Entertainment & Media",
        "expected_key_player": "Warner Bros"
    },
    {
        "text": "Spotify's AI DJ feature, which creates personalized music sets with realistic commentary, has increased listener retention by 25% according to internal data. The company plans to expand the feature with additional voice options and languages.",
        "is_ai_related": True,
        "expected_category": "Entertainment & Media",
        "expected_key_player": "Spotify"
    },
    
    # Non-AI related news articles
    {
        "text": "Apple reported quarterly earnings that exceeded analyst expectations, with iPhone sales driving most of the growth. The company also announced a $90 billion share buyback program and increased its dividend.",
        "is_ai_related": False,
        "expected_category": None,
        "expected_key_player": None
    },
    {
        "text": "The Federal Reserve raised interest rates by 0.25 percentage points, signaling a continued fight against inflation despite recent banking sector stress. Chair Powell indicated that future increases would depend on incoming economic data.",
        "is_ai_related": False,
        "expected_category": None,
        "expected_key_player": None
    },
    {
        "text": "SpaceX successfully launched its Falcon Heavy rocket carrying a classified payload for the U.S. Space Force. The company recovered all three booster stages, marking another milestone for reusable rocket technology.",
        "is_ai_related": False,
        "expected_category": None,
        "expected_key_player": None
    },
    
    # Edge cases and ambiguous content
    {
        "text": "The company's new chip includes neural processing capabilities, though details on specific AI applications remain unclear. Analysts expect the hardware to compete with similar offerings from industry leaders.",
        "is_ai_related": True,
        "expected_category": "Technology & Innovation",
        "expected_key_player": None
    },
    {
        "text": "The rain in Spain falls mainly on the plain, while the snow in Moscow blankets the Kremlin. These weather patterns have persisted throughout history.",
        "is_ai_related": False,
        "expected_category": None,
        "expected_key_player": None
    },
    {
        "text": "A new intelligent design for office chairs has revolutionized workplace comfort. The furniture uses advanced materials science to adjust to different body types.",
        "is_ai_related": False,  # "intelligent" here doesn't refer to AI
        "expected_category": None,
        "expected_key_player": None
    },
    
    # Additional diverse test cases
    {
        "text": "OpenAI and Anthropic have signed the Safe AGI Accord, committing to third-party audits of their most powerful AI systems before deployment. The agreement establishes shared safety standards and information sharing protocols.",
        "is_ai_related": True,
        "expected_category": "Ethics & Social Impact",
        "expected_key_player": "OpenAI and Anthropic"
    },
    {
        "text": "Meta's latest LLM has demonstrated remarkable capabilities in understanding and generating code, outperforming GitHub Copilot on standard benchmarks. The model was trained on trillions of tokens from diverse programming languages.",
        "is_ai_related": True,
        "expected_category": "Research & Development",
        "expected_key_player": "Meta"
    },
    {
        "text": "The Securities and Exchange Commission has launched an investigation into potential market manipulation involving AI-generated news articles about publicly traded companies. The fake articles appeared on legitimate-looking financial news sites.",
        "is_ai_related": True,
        "expected_category": "Policy & Regulation",
        "expected_key_player": "Securities and Exchange Commission"
    },
    {
        "text": "A group of 50 top AI researchers called for a six-month pause on training AI systems more powerful than GPT-4, citing unforeseen risks to society. The open letter has sparked debate about the pace of AI development.",
        "is_ai_related": True,
        "expected_category": "Ethics & Social Impact",
        "expected_key_player": "AI researchers"
    },
    {
        "text": "Google DeepMind and the UK's National Health Service have partnered to develop an AI system for early detection of breast cancer. Initial results show a 20% reduction in false negatives compared to human radiologists alone.",
        "is_ai_related": True,
        "expected_category": "Healthcare & Biotech",
        "expected_key_player": "Google DeepMind"
    },
    {
        "text": "South Korea has announced a $1 billion investment in AI semiconductor development, aiming to reduce dependence on foreign chip suppliers. The initiative includes tax incentives for companies building AI chip fabrication facilities.",
        "is_ai_related": True,
        "expected_category": "Business & Finance",
        "expected_key_player": "South Korea"
    },
    {
        "text": "Universal Music Group has filed a copyright infringement lawsuit against AI music generator Suno, claiming the platform was trained on protected works without permission. The case could set important precedents for generative AI and copyright law.",
        "is_ai_related": True,
        "expected_category": "Entertainment & Media",
        "expected_key_player": "Universal Music Group"
    },
    {
        "text": "The Department of Defense has established new cybersecurity protocols for contractors using AI systems to process classified information. The guidelines mandate specific encryption standards and regular security audits.",
        "is_ai_related": True, 
        "expected_category": "Defense & Security",
        "expected_key_player": "Department of Defense"
    },
    # Business & Finance variations
    {
        "text": "AI startup Scale AI reached unicorn status with its latest funding round, securing $325 million at a $3.5 billion valuation. The company provides data labeling services for machine learning applications.",
        "is_ai_related": True,
        "expected_category": "Business & Finance",
        "expected_key_player": "Scale AI"
    },
    {
        "text": "Intel announced a $10 billion investment in AI chip manufacturing facilities across the United States. The company aims to compete with NVIDIA in the growing market for specialized AI processors.",
        "is_ai_related": True,
        "expected_category": "Business & Finance",
        "expected_key_player": "Intel"
    },
    {
        "text": "Sequoia Capital has launched a dedicated $1 billion fund focused exclusively on AI startups. The venture firm cited unprecedented opportunity in generative AI applications across industries.",
        "is_ai_related": True,
        "expected_category": "Business & Finance",
        "expected_key_player": "Sequoia Capital"
    },
    {
        "text": "Databricks acquired MosaicML for $1.3 billion, expanding its capabilities in AI model training and deployment. The acquisition strengthens Databricks' position against cloud providers in the enterprise AI market.",
        "is_ai_related": True,
        "expected_category": "Business & Finance",
        "expected_key_player": "Databricks"
    },
    {
        "text": "Amazon Web Services reported that AI-related services now account for 30% of its cloud revenue, up from 15% last year. The company has seen particularly strong growth in generative AI applications.",
        "is_ai_related": True,
        "expected_category": "Business & Finance",
        "expected_key_player": "Amazon Web Services"
    },
    
    # Policy & Regulation variations
    {
        "text": "The UK's Competition and Markets Authority has launched an investigation into Microsoft's partnership with OpenAI, examining potential antitrust implications. The probe will assess whether the relationship constitutes a merger.",
        "is_ai_related": True,
        "expected_category": "Policy & Regulation",
        "expected_key_player": "UK's Competition and Markets Authority"
    },
    {
        "text": "California has passed the first state law requiring AI companies to conduct bias audits before deploying systems in high-risk domains like housing and employment. The legislation establishes specific testing standards and documentation requirements.",
        "is_ai_related": True,
        "expected_category": "Policy & Regulation",
        "expected_key_player": "California"
    },
    {
        "text": "The Federal Trade Commission has issued new guidelines on AI-generated content in advertising, requiring clear disclosure when synthetic media is used. Companies must ensure consumers are not misled by realistic AI-generated imagery.",
        "is_ai_related": True,
        "expected_category": "Policy & Regulation",
        "expected_key_player": "Federal Trade Commission"
    },
    {
        "text": "Japan and Singapore have signed a digital trade agreement that includes provisions for AI governance and data sharing. The framework establishes mutual recognition of AI ethics standards and certification procedures.",
        "is_ai_related": True,
        "expected_category": "Policy & Regulation",
        "expected_key_player": "Japan and Singapore"
    },
    {
        "text": "The G7 nations have agreed on a common framework for regulating advanced AI systems, including requirements for risk assessment and management. The accord represents the first multinational approach to AI governance.",
        "is_ai_related": True,
        "expected_category": "Policy & Regulation",
        "expected_key_player": "G7"
    },
    
    # Research & Development variations
    {
        "text": "Scientists at Stanford University have developed a new approach to reinforcement learning that reduces the need for human feedback by 90%. The technique uses synthetic data generation to create diverse training scenarios.",
        "is_ai_related": True,
        "expected_category": "Research & Development",
        "expected_key_player": "Stanford University"
    },
    {
        "text": "A team at Carnegie Mellon has created a breakthrough neural network architecture that can perform complex reasoning tasks with significantly fewer parameters than existing models. The approach uses a novel attention mechanism inspired by human cognitive processes.",
        "is_ai_related": True,
        "expected_category": "Research & Development",
        "expected_key_player": "Carnegie Mellon"
    },
    {
        "text": "Researchers from UC Berkeley published a paper demonstrating that large language models can be compressed by 80% while maintaining 95% of their performance. The technique could make deploying advanced AI more affordable on edge devices.",
        "is_ai_related": True,
        "expected_category": "Research & Development",
        "expected_key_player": "UC Berkeley"
    },
    {
        "text": "OpenAI's latest research paper introduces a novel approach to aligning AI systems with human values through constitutional AI. The method eliminates the need for human feedback on harmful outputs by using the model itself to critique responses.",
        "is_ai_related": True,
        "expected_category": "Research & Development",
        "expected_key_player": "OpenAI"
    },
    {
        "text": "Microsoft Research has developed a multimodal AI system capable of understanding and manipulating 3D environments from natural language instructions. The breakthrough could accelerate progress in robotics and virtual world applications.",
        "is_ai_related": True,
        "expected_category": "Research & Development",
        "expected_key_player": "Microsoft Research"
    },
    
    # Ethics & Social Impact variations
    {
        "text": "A new report from the Brookings Institution examines the potential impact of AI on labor markets, projecting that 25% of jobs could be significantly transformed over the next decade. The study recommends policy interventions to support workforce transitions.",
        "is_ai_related": True,
        "expected_category": "Ethics & Social Impact",
        "expected_key_player": "Brookings Institution"
    },
    {
        "text": "AI Now Institute has published research documenting how automated decision systems in public benefits programs have disproportionately affected low-income communities. The study calls for mandatory algorithmic impact assessments before deployment.",
        "is_ai_related": True,
        "expected_category": "Ethics & Social Impact",
        "expected_key_player": "AI Now Institute"
    },
    {
        "text": "A coalition of teachers' unions has raised concerns about AI-powered proctoring software that monitors students during online exams. They cite privacy violations and high rates of false accusations targeting students of color.",
        "is_ai_related": True,
        "expected_category": "Ethics & Social Impact",
        "expected_key_player": "teachers' unions"
    },
    {
        "text": "Researchers at Princeton found that AI hiring tools from major vendors still demonstrate significant gender and racial bias despite claims of fairness. The systems consistently ranked white male candidates higher for technical positions.",
        "is_ai_related": True,
        "expected_category": "Ethics & Social Impact",
        "expected_key_player": "Princeton"
    },
    {
        "text": "The ACLU has filed a lawsuit challenging the use of facial recognition systems by law enforcement agencies, citing disproportionate error rates for people of color and potential Fourth Amendment violations.",
        "is_ai_related": True,
        "expected_category": "Ethics & Social Impact",
        "expected_key_player": "ACLU"
    },
    
    # Defense & Security variations
    {
        "text": "The U.S. Cyber Command has deployed new AI-based threat detection systems to defend critical infrastructure from state-sponsored attacks. The tools can identify patterns indicative of advanced persistent threats.",
        "is_ai_related": True,
        "expected_category": "Defense & Security",
        "expected_key_player": "U.S. Cyber Command"
    },
    {
        "text": "Israel's military has incorporated AI-powered target recognition in its surveillance drones, allowing for more precise identification of hostile actors. Officials emphasized that lethal force decisions still require human authorization.",
        "is_ai_related": True,
        "expected_category": "Defense & Security",
        "expected_key_player": "Israel's military"
    },
    {
        "text": "China has revealed development of autonomous underwater vehicles equipped with AI for naval reconnaissance missions. The unmanned submarines can operate for months without human intervention.",
        "is_ai_related": True,
        "expected_category": "Defense & Security",
        "expected_key_player": "China"
    },
    {
        "text": "A report from the Center for Strategic and International Studies warns that AI-enabled deepfakes pose an increasing threat to national security, with potential to disrupt elections and military command structures.",
        "is_ai_related": True,
        "expected_category": "Defense & Security",
        "expected_key_player": "Center for Strategic and International Studies"
    },
    {
        "text": "The Defense Advanced Research Projects Agency (DARPA) has launched a $200 million initiative to develop AI systems capable of detecting and neutralizing zero-day cyber vulnerabilities before they can be exploited.",
        "is_ai_related": True,
        "expected_category": "Defense & Security",
        "expected_key_player": "DARPA"
    },
    
    # Technology & Innovation variations
    {
        "text": "Samsung's new smartphone features a dedicated AI neural processing unit that enables advanced computational photography and real-time language translation without internet connectivity.",
        "is_ai_related": True,
        "expected_category": "Technology & Innovation",
        "expected_key_player": "Samsung"
    },
    {
        "text": "Amazon has unveiled a new version of Alexa powered by a large language model, enabling more natural conversations and complex task completion. The assistant can now maintain context across multiple user interactions.",
        "is_ai_related": True,
        "expected_category": "Technology & Innovation",
        "expected_key_player": "Amazon"
    },
    {
        "text": "Intel's latest Xeon processors include specialized matrix multiplication units designed specifically for AI workloads. The company claims 4x performance improvement for machine learning inference compared to previous generations.",
        "is_ai_related": True,
        "expected_category": "Technology & Innovation",
        "expected_key_player": "Intel"
    },
    {
        "text": "Google Maps has implemented a new AI routing algorithm that reduces carbon emissions by optimizing for fuel efficiency. The system analyzes traffic patterns, elevation, and vehicle characteristics to suggest eco-friendly routes.",
        "is_ai_related": True,
        "expected_category": "Technology & Innovation",
        "expected_key_player": "Google Maps"
    },
    {
        "text": "Adobe has integrated generative AI across its Creative Cloud suite, allowing designers to generate and edit images from text prompts. The tools include built-in attribution for AI-generated content to address copyright concerns.",
        "is_ai_related": True,
        "expected_category": "Technology & Innovation",
        "expected_key_player": "Adobe"
    },
    
    # Healthcare & Biotech variations
    {
        "text": "Researchers at Johns Hopkins have developed an AI system that can predict patient deterioration in hospital settings 24 hours before critical events occur. The algorithm analyzes subtle patterns in vital signs and lab results.",
        "is_ai_related": True,
        "expected_category": "Healthcare & Biotech",
        "expected_key_player": "Johns Hopkins"
    },
    {
        "text": "Novartis has partnered with Microsoft to use AI for accelerating drug discovery, applying machine learning to predict which compounds are likely to successfully treat specific diseases.",
        "is_ai_related": True,
        "expected_category": "Healthcare & Biotech",
        "expected_key_player": "Novartis"
    },
    {
        "text": "A new AI-powered hearing aid can selectively amplify human speech while filtering out background noise, significantly improving comprehension in crowded environments for people with hearing loss.",
        "is_ai_related": True,
        "expected_category": "Healthcare & Biotech",
        "expected_key_player": None
    },
    {
        "text": "Stanford Medical Center has implemented an AI system for analyzing medical imaging across all departments. The tool has increased diagnosis speed by 30% while reducing radiologist workload.",
        "is_ai_related": True,
        "expected_category": "Healthcare & Biotech",
        "expected_key_player": "Stanford Medical Center"
    },
    {
        "text": "The National Institutes of Health has launched a $100 million initiative to develop AI tools for rare disease diagnosis. The program aims to reduce the typical seven-year diagnostic odyssey faced by patients.",
        "is_ai_related": True,
        "expected_category": "Healthcare & Biotech",
        "expected_key_player": "National Institutes of Health"
    },
    
    # Entertainment & Media variations
    {
        "text": "Disney has revealed that its upcoming animated feature utilized AI tools for background generation and character animation refinement. Artists emphasized that the technology augmented rather than replaced human creativity.",
        "is_ai_related": True,
        "expected_category": "Entertainment & Media",
        "expected_key_player": "Disney"
    },
    {
        "text": "Universal Music has launched an AI voice clone service allowing artists to license digital versions of their voices for approved commercial use. The technology includes safeguards to prevent unauthorized impersonation.",
        "is_ai_related": True,
        "expected_category": "Entertainment & Media",
        "expected_key_player": "Universal Music"
    },
    {
        "text": "The New York Times has implemented an AI content moderation system for its comments section, significantly reducing the publication delay for reader contributions while maintaining community standards.",
        "is_ai_related": True,
        "expected_category": "Entertainment & Media",
        "expected_key_player": "New York Times"
    },
    {
        "text": "Epic Games has incorporated machine learning into its Unreal Engine, enabling game developers to create more realistic character behaviors and environmental interactions without complex programming.",
        "is_ai_related": True,
        "expected_category": "Entertainment & Media",
        "expected_key_player": "Epic Games"
    },
    {
        "text": "TikTok's recommendation algorithm has been updated with advanced AI capabilities to better match content with user interests. The changes have increased average time spent on the platform by 20%.",
        "is_ai_related": True,
        "expected_category": "Entertainment & Media",
        "expected_key_player": "TikTok"
    },
    
    # Non-AI related variations
    {
        "text": "Gold prices reached a record high today as investors seek safe haven assets amid growing economic uncertainty. Analysts predict continued strength in precious metals through year-end.",
        "is_ai_related": False,
        "expected_category": None,
        "expected_key_player": None
    },
    {
        "text": "The European Central Bank maintained interest rates despite rising inflation, citing concerns about economic growth. Bank President Christine Lagarde indicated a willingness to act if price pressures persist.",
        "is_ai_related": False,
        "expected_category": None,
        "expected_key_player": None
    },
    {
        "text": "Hurricane Laura has strengthened to a Category 4 storm as it approaches the Gulf Coast. Authorities have ordered mandatory evacuations for vulnerable coastal communities.",
        "is_ai_related": False,
        "expected_category": None,
        "expected_key_player": None
    },
    {
        "text": "The Senate passed a bipartisan infrastructure bill allocating $1 trillion for roads, bridges, broadband and other projects. The legislation now moves to the House for consideration.",
        "is_ai_related": False,
        "expected_category": None,
        "expected_key_player": None
    },
    {
        "text": "Scientists have discovered a previously unknown species of deep-sea fish in the Mariana Trench. The bioluminescent creature has adapted to extreme pressure conditions.",
        "is_ai_related": False,
        "expected_category": None,
        "expected_key_player": None
    },
    
    # Edge cases and ambiguous content
    {
        "text": "The company unveiled its new 'intelligent' manufacturing process, which uses robotics and automation to increase efficiency. The system reduced production costs by 15%.",
        "is_ai_related": False,  # 'intelligent' here refers to automation, not AI
        "expected_category": None,
        "expected_key_player": None
    },
    {
        "text": "The robot vacuum cleaner uses machine vision to navigate around obstacles and create a map of your home. It features advanced suction technology and a self-emptying dustbin.",
        "is_ai_related": True,  # machine vision is a form of AI
        "expected_category": "Technology & Innovation",
        "expected_key_player": None
    },
    {
        "text": "Neural networks in the brain are responsible for complex cognitive functions like memory and learning. Neuroscientists have mapped new connections between the hippocampus and prefrontal cortex.",
        "is_ai_related": False,  # biological neural networks, not AI
        "expected_category": None,
        "expected_key_player": None
    },
    {
        "text": "The smart thermostat learns your preferences over time and adjusts temperature settings automatically. It can be controlled via smartphone app or voice commands.",
        "is_ai_related": True,  # learning preferences implies machine learning
        "expected_category": "Technology & Innovation",
        "expected_key_player": None
    },
    {
        "text": "Researchers have developed a computer model to predict weather patterns with greater accuracy. The system analyzes historical data and current conditions to generate forecasts.",
        "is_ai_related": True,  # predictive modeling is a form of AI
        "expected_category": "Research & Development",
        "expected_key_player": None
    }
]

def run_classifier_tests():
    """Run all classification tests and report results"""
    results = []
    scores = {
        "ai_related": {"correct": 0, "total": 0},
        "category": {"correct": 0, "total": 0},
        "key_player": {"correct": 0, "total": 0}
    }
    
    for i, case in enumerate(TEST_CASES):
        print(f"Testing case {i+1}/{len(TEST_CASES)}", end="\r")
        
        # Test AI relevance
        ai_related, ai_confidence = is_ai_related(case["text"])
        is_ai_correct = ai_related == case["is_ai_related"]
        scores["ai_related"]["total"] += 1
        if is_ai_correct:
            scores["ai_related"]["correct"] += 1
            
        # Test category assignment (only if AI-related)
        category, category_confidence = (None, 0)
        is_category_correct = False
        if case["is_ai_related"]:
            category, category_confidence = determine_category(case["text"])
            is_category_correct = category == case["expected_category"]
            scores["category"]["total"] += 1
            if is_category_correct:
                scores["category"]["correct"] += 1
                
        # Test key player extraction (only if AI-related)
        key_player, player_confidence = (None, 0)
        is_player_correct = False
        if case["is_ai_related"] and case["expected_key_player"]:
            key_player, player_confidence = extract_key_player(case["text"])
            # More flexible matching for key player (partial match is acceptable)
            if key_player and case["expected_key_player"]:
                is_player_correct = (key_player.lower() in case["expected_key_player"].lower() or 
                                    case["expected_key_player"].lower() in key_player.lower())
            else:
                is_player_correct = key_player == case["expected_key_player"]
                
            scores["key_player"]["total"] += 1
            if is_player_correct:
                scores["key_player"]["correct"] += 1
        
        # Calculate overall confidence
        overall_confidence = 0
        if case["is_ai_related"]:
            overall_confidence = (0.4 * ai_confidence) + (0.6 * category_confidence)
        else:
            overall_confidence = ai_confidence
            
        # Store result
        results.append({
            "text_preview": case["text"][:50] + "..." if len(case["text"]) > 50 else case["text"],
            "is_ai_related": {
                "expected": case["is_ai_related"],
                "actual": ai_related,
                "correct": is_ai_correct,
                "confidence": ai_confidence
            },
            "category": {
                "expected": case["expected_category"],
                "actual": category,
                "correct": is_category_correct,
                "confidence": category_confidence
            },
            "key_player": {
                "expected": case["expected_key_player"],
                "actual": key_player,
                "correct": is_player_correct,
                "confidence": player_confidence
            },
            "overall_confidence": overall_confidence
        })
    
    return results, scores

def print_test_results(results, scores):
    """Print test results in a formatted table"""
    print("\n" + "="*100)
    print(colored("AI WORLD PRESS CLASSIFIER TEST RESULTS", "cyan", attrs=["bold"]))
    print("="*100)
    
    # Prepare results table
    table_data = []
    for i, result in enumerate(results):
        ai_result = "✓" if result["is_ai_related"]["correct"] else "✗"
        ai_result_color = "green" if result["is_ai_related"]["correct"] else "red"
        
        category_result = "✓" if result["category"]["correct"] else "✗"
        category_result_color = "green" if result["category"]["correct"] else "red"
        
        key_player_result = "✓" if result["key_player"]["correct"] else "✗"
        key_player_result_color = "green" if result["key_player"]["correct"] else "red"
        
        table_data.append([
            i+1,
            result["text_preview"],
            colored(f"{ai_result} ({result['is_ai_related']['expected']})", ai_result_color),
            f"{result['is_ai_related']['confidence']:.2f}",
            colored(f"{category_result} ({result['category']['expected'] or 'None'})", category_result_color) 
                if result["is_ai_related"]["expected"] else "N/A",
            f"{result['category']['confidence']:.2f}" if result["is_ai_related"]["expected"] else "N/A",
            colored(f"{key_player_result} ({result['key_player']['expected'] or 'None'})", key_player_result_color)
                if result["is_ai_related"]["expected"] and result["key_player"]["expected"] else "N/A",
            f"{result['key_player']['confidence']:.2f}" if result["is_ai_related"]["expected"] and result["key_player"]["expected"] else "N/A",
            f"{result['overall_confidence']:.2f}"
        ])
    
    # Print results table
    print(tabulate.tabulate(
        table_data,
        headers=[
            "ID", "Text Preview", "AI Related", "Conf.", 
            "Category", "Conf.", "Key Player", "Conf.", "Overall Conf."
        ],
        tablefmt="pretty"
    ))
    
    # Calculate and print accuracy scores
    ai_accuracy = scores["ai_related"]["correct"] / scores["ai_related"]["total"] * 100
    category_accuracy = scores["category"]["correct"] / scores["category"]["total"] * 100 if scores["category"]["total"] > 0 else 0
    key_player_accuracy = scores["key_player"]["correct"] / scores["key_player"]["total"] * 100 if scores["key_player"]["total"] > 0 else 0
    
    overall_correct = scores["ai_related"]["correct"] + scores["category"]["correct"] + scores["key_player"]["correct"]
    overall_total = scores["ai_related"]["total"] + scores["category"]["total"] + scores["key_player"]["total"]
    overall_accuracy = overall_correct / overall_total * 100
    
    print("\n" + "-"*100)
    print(colored("ACCURACY SUMMARY", "yellow", attrs=["bold"]))
    print("-"*100)
    
    accuracy_table = [
        ["AI Relevance Detection", f"{scores['ai_related']['correct']} / {scores['ai_related']['total']}", f"{ai_accuracy:.2f}%"],
        ["Topic Categorization", f"{scores['category']['correct']} / {scores['category']['total']}", f"{category_accuracy:.2f}%"],
        ["Key Player Extraction", f"{scores['key_player']['correct']} / {scores['key_player']['total']}", f"{key_player_accuracy:.2f}%"],
        ["OVERALL", f"{overall_correct} / {overall_total}", f"{overall_accuracy:.2f}%"]
    ]
    
    print(tabulate.tabulate(
        accuracy_table,
        headers=["Task", "Score", "Accuracy"],
        tablefmt="pretty"
    ))
    
    # Print performance assessment
    print("\n" + "-"*100)
    print(colored("PERFORMANCE ASSESSMENT", "yellow", attrs=["bold"]))
    print("-"*100)
    
    for task, accuracy in [
        ("AI Relevance Detection", ai_accuracy),
        ("Topic Categorization", category_accuracy),
        ("Key Player Extraction", key_player_accuracy),
        ("Overall", overall_accuracy)
    ]:
        if accuracy >= 90:
            status = colored("EXCELLENT", "green", attrs=["bold"])
        elif accuracy >= 80:
            status = colored("GOOD", "blue", attrs=["bold"])
        elif accuracy >= 70:
            status = colored("ACCEPTABLE", "yellow", attrs=["bold"])
        else:
            status = colored("NEEDS IMPROVEMENT", "red", attrs=["bold"])
            
        print(f"{task}: {status} ({accuracy:.2f}%)")

if __name__ == "__main__":
    print("Running AI World Press classifier tests...")
    results, scores = run_classifier_tests()
    print_test_results(results, scores)