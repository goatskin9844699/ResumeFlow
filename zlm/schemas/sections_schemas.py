'''
-----------------------------------------------------------------------
File: schemas/sections_schmas.py
Creation Time: Aug 18th 2024, 2:26 am
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2024 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
'''

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, ValidationError, validator
import json
from datetime import datetime

def validate_date_format(v: str) -> str:
    """Validates that the date string is in MM/YYYY format or 'Present'."""
    if v == "Present":
        return v
    try:
        datetime.strptime(v, "%m/%Y")
        return v
    except ValueError:
        raise ValueError("Date must be in MM/YYYY format (e.g., 08/2023) or 'Present'")

def format_validation_error(error: ValidationError, json_str: str) -> str:
    """
    Formats validation errors in a human-readable way.
    
    Args:
        error: Pydantic ValidationError
        json_str: The JSON string being validated
        
    Returns:
        str: Formatted error message
    """
    unique_errors = set()
    
    for err in error.errors():
        loc = err['loc']
        field_name = str(loc[-1])
        
        # Create error message
        if err['msg'] == 'field required':
            msg = f"Missing required field '{field_name}'"
            if len(loc) > 1:
                parent = loc[-2]
                if isinstance(parent, int):
                    parent = loc[-3] if len(loc) > 2 else 'item'
                msg += f" in {parent}"
        else:
            msg = f"Invalid value for field '{field_name}': {err['msg']}"
        
        unique_errors.add(msg)
    
    if not unique_errors:
        return "No validation errors found."
    
    return "Invalid resume data structure:\n\n" + "\n\n".join(sorted(unique_errors))

class Achievements(BaseModel):
    achievements: List[str] = Field(description="job relevant key accomplishments, awards, or recognitions that demonstrate your skills and abilities.")

class Publication(BaseModel):
    authors: str = Field(description="The authors of the publication, with main author in bold if applicable.")
    title: str = Field(description="The title of the publication.")
    location: str = Field(description="The venue or location where the publication was presented/published.")
    date: str = Field(description="The date of publication or presentation in YYYY format.")

class Publications(BaseModel):
    publications: List[Publication] = Field(description="Academic or professional publications, including authors, title, venue, and date.")

class Certification(BaseModel):
    name: str = Field(description="The name of the certification.")
    by: str = Field(description="The organization or institution that issued the certification.")
    link: str = Field(description="A link to verify the certification.")

class Certifications(BaseModel):
    certifications: List[Certification] = Field(description="job relevant certifications that you have earned, including the name, issuing organization, and a link to verify the certification.")

class Education(BaseModel):
    degree: str = Field(description="The degree or qualification obtained and The major or field of study. e.g., Bachelor of Science in Computer Science.")
    university: str = Field(description="The name of the institution where the degree was obtained with location. e.g. Arizona State University, Tempe, USA")
    from_date: str = Field(description="The start date of the education period in MM/YYYY format. e.g., 08/2023")
    to_date: str = Field(description="The end date of the education period in MM/YYYY format. e.g., 05/2025")
    courses: List[str] = Field(description="Relevant courses or subjects studied during the education period. e.g. [Data Structures, Algorithms, Machine Learning]")

    @validator('from_date', 'to_date')
    def validate_dates(cls, v):
        return validate_date_format(v)

class Educations(BaseModel):
    education: List[Education] = Field(description="Educational qualifications, including degree, institution, dates, and relevant courses.")

class Project(BaseModel):
    name: str = Field(description="The name or title of the project.")
    type: str | None = Field(description="The type or category of the project, such as hackathon, publication, professional, and academic.")
    link: str = Field(description="A link to the project repository or demo.")
    from_date: str = Field(description="The start date of the project in MM/YYYY format. e.g., 08/2023")
    to_date: str = Field(description="The end date of the project in MM/YYYY format. e.g., 11/2023")
    description: List[str] = Field(description="A list of 3 bullet points describing the project experience, tailored to match job requirements. Each bullet point should follow the 'Did X by doing Y, achieved Z' format, quantify impact, implicitly use STAR methodology, use strong action verbs, and be highly relevant to the specific job. Ensure clarity, active voice, and impeccable grammar.")

    @validator('from_date', 'to_date')
    def validate_dates(cls, v):
        return validate_date_format(v)

class Projects(BaseModel):
    projects: List[Project] = Field(description="Project experiences, including project name, type, link, dates, and description.")

class SkillSection(BaseModel):
    name: str = Field(description="name or title of the skill group and competencies relevant to the job, such as programming languages, data science, tools & technologies, cloud & DevOps, full stack,  or soft skills.")
    skills: List[str] = Field(description="Specific skills or competencies within the skill group, such as Python, JavaScript, C#, SQL in programming languages.")

class SkillSections(BaseModel):
    skill_section: List[SkillSection] = Field(description="Skill sections, each containing a group of skills and competencies relevant to the job.")

class Experience(BaseModel):
    role: str = Field(description="The job title or position held. e.g. Software Engineer, Machine Learning Engineer.")
    company: str = Field(description="The name of the company or organization.")
    location: str = Field(description="The location of the company or organization. e.g. San Francisco, USA.")
    from_date: str = Field(description="The start date of the employment period in MM/YYYY format. e.g., 08/2023")
    to_date: str = Field(description="The end date of the employment period in MM/YYYY format. e.g., 11/2025")
    description: List[str] = Field(description="A list of 3 bullet points describing the work experience, tailored to match job requirements. Each bullet point should follow the 'Did X by doing Y, achieved Z' format, quantify impact, implicitly use STAR methodology, use strong action verbs, and be highly relevant to the specific job. Ensure clarity, active voice, and impeccable grammar.")

    @validator('from_date', 'to_date')
    def validate_dates(cls, v):
        return validate_date_format(v)

class Experiences(BaseModel):
    work_experience: List[Experience] = Field(description="Work experiences, including job title, company, location, dates, and description.")

class Media(BaseModel):
    linkedin: Optional[HttpUrl] = Field(description="LinkedIn profile URL", default=None)
    github: Optional[HttpUrl] = Field(description="GitHub profile URL", default=None)
    medium: Optional[HttpUrl] = Field(description="Medium profile URL", default=None)
    devpost: Optional[HttpUrl] = Field(description="Devpost profile URL", default=None)

class ResumeSchema(BaseModel):
    name: str = Field(description="The full name of the candidate.")
    summary: Optional[str] = Field(description="A brief summary or objective statement highlighting key skills, experience, and career goals.")
    phone: str = Field(description="The contact phone number of the candidate.")
    email: str = Field(description="The contact email address of the candidate.")
    media: Media = Field(description="Links to professional social media profiles, such as LinkedIn, GitHub, or personal website.")
    work_experience: List[Experience] = Field(description="Work experiences, including job title, company, location, dates, and description.")
    education: List[Education] = Field(description="Educational qualifications, including degree, institution, dates, and relevant courses.")
    skill_section: List[SkillSection] = Field(description="Skill sections, each containing a group of skills and competencies relevant to the job.")
    projects: List[Project] = Field(description="Project experiences, including project name, type, link, dates, and description.")
    publications: Optional[List[Publication]] = Field(description="Academic or professional publications, including authors, title, venue, and date.", default=[])
    certifications: List[Certification] = Field(description="job relevant certifications that you have earned, including the name, issuing organization, and a link to verify the certification.")
    achievements: List[str] = Field(description="job relevant key accomplishments, awards, or recognitions that demonstrate your skills and abilities.")

    @classmethod
    def validate_json(cls, json_str: str) -> Dict[str, Any]:
        """
        Validates JSON string against the schema.
        
        Args:
            json_str: The JSON string to validate
            
        Returns:
            Dict[str, Any]: Validated data if successful
            
        Raises:
            ValueError: If validation fails, with formatted error messages
        """
        try:
            data = json.loads(json_str)
            return cls(**data)
        except ValidationError as e:
            error_msg = format_validation_error(e, json_str)
            raise ValueError(error_msg)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e.msg}")